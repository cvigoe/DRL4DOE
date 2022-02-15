"""
DRL4DOE Gym Env

Author: Conor Igoe
Date: 02/09/2022
"""
import pudb
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torch

class DRL4DOE(gym.Env):
    """Initialises the DRL4DOE toy problem
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_test_points=50):
        super(DRL4DOE, self).__init__()
        self.num_test_points = num_test_points
        self.action_space = spaces.Box( np.array([0]) , np.array([num_test_points]) )
        diag_dim = self.num_test_points*2
        self.observation_space = spaces.Box(   
            np.array([-1*np.inf]*diag_dim),    
            np.array([np.inf]*diag_dim))

    def initialise_environment(self, env_str, noise_sigma=.1, T=30, 
        region_length=20, ucb=False, NUM_MC_ITERS=500):
        self.test_points = np.linspace(0,region_length,
            self.num_test_points)
        self.noise_sigma = noise_sigma
        self.T = T
        self.ucb = ucb
        self.NUM_MC_ITERS = NUM_MC_ITERS

    def reset(self):
        """Resets the gym environment, redrawing ground truth

        Returns:
          state: 
            numpy array of prior mean concatenated with flattened 
            upper triangular entries of covariance matrix
        """        
        self.GP = GP(zero_mean, SE_kernel, initial_dataset=None,
            sigma2_n=self.noise_sigma**2)
        mu, Sig = self.GP.calculate_prior_mean_cov(self.test_points)
        self.ground_truth = self.GP.draw(mu, Sig)
        self.t = 0
        diag_Sig = np.diag(Sig)
        return np.concatenate([mu, np.log(diag_Sig) ])

    def step(self, action):
        """Takes a step in the DRL4DOE toy problem.

        Args:
            action:
              The point to evaluate the GP

        Returns:

          A tuple (state, reward, done, info). The state variable is 
          the mean vector and the covariance matrix at all test 
          points. The reward is the noisy observation. Task isdone 
          when we exceed self.T budget.

        """
        p = np.random.rand() 
        if self.ucb and (p < 0.2):
            mu, Sig = self.GP.calculate_posterior_mean_cov(
                self.test_points)
            delta = 0.95
            beta_t = 2*np.log( (self.t+1)**(5/2) * (np.pi**2) / (3*delta)) # GP-UCB formula 
            action = np.argmax(mu + np.sqrt(beta_t)*np.sqrt(np.diag(Sig)))
        else:
            action /= 2
            action += 0.5
            action = int(np.clip(action*self.num_test_points, 0, self.num_test_points - 0.005))

        observation = self.ground_truth[action] + \
        np.random.randn()*self.noise_sigma
        self.GP.add_observations(np.asarray([[self.test_points[action],observation]]))
        self.t += 1
        mu, Sig = self.GP.calculate_posterior_mean_cov(
            self.test_points)
        diag_Sig = np.diag(Sig)

        reward = 0
        for i in range(self.NUM_MC_ITERS):
            draw = self.GP.draw(mu, Sig)
            inst_regret = max(draw) - mu[action]
            reward -= inst_regret/self.NUM_MC_ITERS

        # return (np.concatenate([mu,np.log(diag_Sig)]),
        # -1*np.array(max(self.ground_truth) -
        #     self.ground_truth[action]),
        # np.array(self.t > self.T) , {})

        return (np.concatenate([mu,np.log(diag_Sig)]),
        np.array(reward),
        np.array(self.t > self.T),
        {})

class GP():
    def __init__(self, mean_function, kernel, initial_dataset, 
        eps=0.00001, sigma2_n=1):
        self.mean_function = mean_function
        self.kernel = kernel
        self.dataset = initial_dataset
        self.eps = eps
        self.sigma2_n = sigma2_n

    def get_next_UCB_query(self, test_points, t, delta=0.95):
        beta_t = 2* np.log( t**(5/2) * (np.pi**2) / (3*delta))         # GP-UCB formula from 10-403 slides
        if t == 0:
            mu, Sig = self.calculate_prior_mean_cov(test_points)
        else:
            mu, Sig = self.calculate_posterior_mean_cov(test_points)
        return test_points[
        np.argmax(mu + np.sqrt(beta_t)*np.diag(Sig))]


    def calculate_prior_mean_cov(self, test_points):
        X_test = test_points

        mu_test_prior = self.mean_function(X_test)
        Sigma_test_test_prior = self.kernel(X_test, X_test)
        return mu_test_prior, Sigma_test_test_prior

    def calculate_posterior_mean_cov(self, test_points):
        if self.dataset is None:
            return self.calculate_prior_mean_cov(test_points)
        X_train = self.dataset[:,0]
        y_train = self.dataset[:,1]

        X_test = test_points

        mu_train_prior = self.mean_function(X_train)
        mu_test_prior = self.mean_function(X_test)

        Sigma_train_train_prior = self.kernel(X_train, X_train)
        Sigma_train_test_prior = self.kernel(X_train, X_test)
        Sigma_test_test_prior = self.kernel(X_test, X_test)

        L = np.linalg.cholesky(Sigma_train_train_prior + \
            self.sigma2_n*np.eye(*Sigma_train_train_prior.shape))

        mu_test_posterior = mu_test_prior + \
        Sigma_train_test_prior.T @ \
        np.linalg.solve(L.T, np.linalg.solve(L,y_train))
        Sigma_test_test_posterior = Sigma_test_test_prior - \
        Sigma_train_test_prior.T @ \
        np.linalg.solve(L.T, 
            np.linalg.solve(L,Sigma_train_test_prior))
        
        return mu_test_posterior, Sigma_test_test_posterior

    def draw(self, mu, Sig):
        L = np.linalg.cholesky(Sig + self.eps*np.eye(*Sig.shape))
        u = np.random.normal(loc=0, scale=1, size=L.shape[0])
        return mu + np.dot(L, u)

    def add_observations(self, observations):
        if self.dataset is None:
            self.dataset = observations
        else:
            self.dataset = np.concatenate(
                (self.dataset, observations))

def SE_kernel(X1, X2, sigma2=1, ell=1):
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for index1, x1 in enumerate(X1):
        for index2, x2 in enumerate(X2):
            K[index1, index2] = sigma2*np.exp(
                -(np.asarray(x1) - np.asarray(x2))**2 / (2*(ell**2)))
    return K

def zero_mean(x):
    return np.zeros(x.shape)

