"""
Simple toy environments to test on.

Author: Ian Char
Date: 06/28/2021
"""
from functools import partial

import gym
import numpy as np

from gym_drldoe.envs.drl4doe_env import GP, zero_mean, SE_kernel


class RandomGridGPEnv(gym.Env):
    """
    A BO environment with a grid on a GP. The grid is such that
    with an oracle agent, monotonic rewards can be achieved.
    """

    def __init__(
            self,
            joint_info=True,
            act_dim=10,
            horizon=float('inf'),
            noise_sigma=1e-3,
            sample_fidelity=250,
            region_length=1,
            length_scale=0.1,
            length_scale_prior_bounds=None,
            relu_improvement_reward=False,
            time_in_state=False,
            rew_scale=100,
    ):
        """Constructor:
        Args:
            joint_information: Whether to have joint information
                about the environment or just marginal info.
            horizon: How long we can roll out before returning a terminal.
            act_dim: The number of options the agent has every
                iteration. This corresponds to number of grid pts.
            noise_sigma: Standard deviation of the noise.
            sample_fidelity: The fidelity of the function drawn.
            region_length: How long the X region should be.
            length_scale: Length scale parameter for the SE_kernel.
            length_scale_prior_bounds: Optional tuple of lower and upper bound
                for uniform prior over length scale e.g. (0.1, 0.3)
            relu_improvement_reward: Whether reward should be relu improvement
                over the best seen.
            time_in_state: Whether the time should be included into state.
                If set to True, append to end of state.
            rew_scale: How much to scale the rewards by.
        """
        super().__init__()
        self._joint_info = joint_info
        obs_dim = act_dim * 2 + time_in_state
        if joint_info:
            obs_dim += int(act_dim * (act_dim - 1) / 2)
        self.observation_space = gym.spaces.Box(
            # Note actually bounded by -1 and 1 but whatever.
            -1 * np.ones(obs_dim),
            np.ones(obs_dim),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(act_dim)
        self.act_dim = act_dim
        self._horizon = horizon
        self._length_scale = length_scale
        self._length_scale_prior_bounds = length_scale_prior_bounds
        self._noise_sigma = noise_sigma
        self._relu_improvement_reward = relu_improvement_reward
        self._time_in_state = time_in_state
        self._rew_scale = rew_scale
        self._kernel = None
        self._gp = None
        self._ground_truth = None
        self._sample_max_val = None
        self._sample_min_val = None
        self._best_found = None
        self._grid_idxs = None
        self._t = 0
        self._function_grid = np.linspace(0, region_length, sample_fidelity)
        self._sample_fidelity = sample_fidelity

    def reset(self, seed=None):
        """Reset the environment and return the state."""
        np.random.seed(seed)
        self._new_gp()
        mu, sigma = self._gp.calculate_prior_mean_cov(self._function_grid)
        self._ground_truth = self._gp.draw(mu, sigma)
        self._sample_max_val = np.max(self._ground_truth)
        self._sample_min_val = np.min(self._ground_truth)
        self._t = 0
        self._best_found = self._sample_min_val
        return self._get_obs(self._get_new_grid())

    def step(self, action):
        assert action < len(self._grid_idxs)
        val = self._ground_truth[self._grid_idxs[action]]
        xpt = self._function_grid[self._grid_idxs[action]]
        obs = val + np.random.randn() * self._noise_sigma
        self._gp.add_observations(np.array([[xpt, obs]]))
        self._t += 1
        nxt = self._get_obs(self._get_new_grid())
        if self._relu_improvement_reward:
            rew = max(0, (val - self._best_found)
                          / (self._sample_max_val - self._sample_min_val))
        else:
            rew = val - self._sample_max_val
        rew *= self._rew_scale
        self._best_found = max(self._best_found, val)
        done = self._t >= self._horizon
        return nxt, rew, done, {}

    def _get_obs(self, grid):
        """Get observation for the grid."""
        mu, sigma = self._gp.calculate_posterior_mean_cov(grid)
        if self._joint_info:
            var_info = sigma[np.triu_indices(sigma.shape[0])]
        else:
            var_info = sigma.diagonal()
        obs = np.concatenate([mu, var_info])
        if self._time_in_state:
            obs = np.append(obs, self._t)
        return obs

    def _get_new_grid(self):
        if self._grid_idxs is None:
            self._grid_idxs = np.random.choice(
                np.arange(self._sample_fidelity),
                size=self.act_dim,
                replace=False,
            )
        else:
            best_grid_idx = np.argmax(self._ground_truth[self._grid_idxs])
            best_grid_pt = self._grid_idxs[best_grid_idx]
            new_pool = np.append(
                np.arange(0, best_grid_pt),
                np.arange(best_grid_pt + 1, self._sample_fidelity),
            )
            self._grid_idxs = np.append(best_grid_pt, np.random.choice(
                new_pool,
                size=self.act_dim - 1,
                replace=False,
            ))
            np.random.shuffle(self._grid_idxs)
        return self._function_grid[self._grid_idxs]

    def _new_gp(self):
        if self._length_scale_prior_bounds is not None:
            self._length_scale = np.random.uniform(
                self._length_scale_prior_bounds[0],
                self._length_scale_prior_bounds[1],
            )
        self._kernel = partial(SE_kernel, ell=self._length_scale)
        self._gp = GP(zero_mean, self._kernel, initial_dataset=None,
                      sigma2_n=self._noise_sigma)


    def render(self):
        pass
