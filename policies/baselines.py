"""
List of policies given discrete action choices.
"""
import abc

import numpy as np
import torch

from rlkit.policies.argmax import ArgmaxDiscretePolicy


class Policy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_action(self, state):
        """Take in the current state and return tuple of (action, info)"""

    def reset(self):
        """Reset the policy."""
        pass


class RlPolicy(Policy):

    def __init__(self, path):
        loaded = torch.load(path)
        self.policy = ArgmaxDiscretePolicy(loaded['trainer/qf'].cpu())

    def get_action(self, state):
        return self.policy.get_action(state)


class RandomPolicy(Policy):

    def __init__(self, act_dim):
        self.act_dim = act_dim

    def get_action(self, state):
        return np.random.randint(self.act_dim), {}


class UCBPolicy(Policy):

    def __init__(self, act_dim, delta=0.95):
        self.act_dim = act_dim
        self.delta = delta
        self.t = 0

    def reset(self):
        self.t = 0

    def get_action(self, state):
        self.t += 1
        beta_t = 2* np.log( self.t**(5/2) * (np.pi**2) / (3*self.delta))
        mu = state[:self.act_dim]
        vrs = state[self.act_dim:2 * self.act_dim]
        return np.argmax(mu + np.sqrt(beta_t) * vrs), {'beta': beta_t}
