"""
Evaluate a policy on an environment.

Author: Ian Char
Date: June 29, 2021
"""
import argparse

import h5py
import numpy as np
import torch

from gym_drldoe.envs.simple_gp_envs import RandomGridGPEnv
from rlkit.policies.argmax import ArgmaxDiscretePolicy


def load_policy(args):
    loaded = torch.load(args.rl_policy_path)
    return ArgmaxDiscretePolicy(loaded['trainer/qf'])

def evaluate(args):
    if policy_type == 'rl':
        policy = load_policy(args)
    elif policy_type == 'ucb':
        raise NotImplementedError('TODO')
        # policy = ucb_policy
    env = RandomGridGPEnv(joint_info=args.joint_info)
    trajs = []
    for seed in range(args.num_evals):
        regrets = []
        state = env.reset(seed=seed)
        for _ in range(args.episode_length):
            act = policy.get_action(state)
            state, reg, _, _ = env.step(act)
            regrets.append(reg)
        trajs.append(np.array(regrets))
    trajs = np.array(trajs)
    with h5py.File(args.save_path, 'w') as hdata:
        hdata.create_dataset('regrets', data=trajs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--policy_type', required=True, choice=['rl', 'ucb'])
    parser.add_argument('--rl_policy_path')
    parser.add_argument('--joint_info', action='store_true')
    parser.add_argument('--episode_length', type=int, default=50)
    parser.add_argument('--num_evals', type=int, default=100)
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()
