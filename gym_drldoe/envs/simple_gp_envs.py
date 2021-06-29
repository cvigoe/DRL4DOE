"""
Simple toy environments to test on.

Author: Ian Char
Date: 06/28/2021
"""
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
            noise_sigma=1,
            sample_fidelity=250,
            region_length=1,
    ):
        """Constructor:
        Args:
            joint_information: Whether to have joint information
                about the environment or just marginal info.
            action_dim: The number of options the agent has every
                iteration. This corresponds to number of grid pts.
            noise_sigma: Standard deviation of the noise.
            sample_fidelity: The fidelity of the function drawn.
        """
        super().__init__()
        self._joint_info = joint_info
        obs_dim = act_dim * 2
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
        self._noise_sigma = noise_sigma
        self._gp = GP(zero_mean, SE_kernel, initial_dataset=None)
        self._state = None
        self._ground_truth = None
        self._sample_max_val = None
        self._grid_idxs = None
        self._t = 0
        self._function_grid = np.linspace(0, region_length, sample_fidelity)
        self._sample_fidelity = sample_fidelity

    def reset(self, seed=None):
        """Reset the environment and return the state."""
        np.random.seed(seed)
        self._gp = GP(zero_mean, SE_kernel, initial_dataset=None)
        mu, sigma = self._gp.calculate_prior_mean_cov(self._function_grid)
        self._ground_truth = self._gp.draw(mu, sigma)
        self._sample_max_val = np.max(self._ground_truth)
        self._t = 0
        return self._get_obs(self._get_new_grid())

    def step(self, action):
        assert action < len(self._grid_idxs)
        val = self._ground_truth[self._grid_idxs[action]]
        xpt = self._function_grid[self._grid_idxs[action]]
        obs = val + np.random.randn() * self._noise_sigma
        self._gp.add_observations(np.array([[xpt, obs]]))
        self._t += 1
        nxt = self._get_obs(self._get_new_grid())
        rew = val - self._sample_max_val
        done = False
        return nxt, rew, done, {}

    def _get_obs(self, grid):
        """Get observation for the grid."""
        mu, sigma = self._gp.calculate_posterior_mean_cov(grid)
        if self._joint_info:
            var_info = sigma[np.triu_indices(sigma.shape[0])]
        else:
            var_info = sigma.diagonal()
        return np.concatenate([mu, var_info])

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

    def render(self):
        pass
