"""
Evaluate a policy on an environment.

Author: Ian Char
Date: June 29, 2021
"""
import argparse

import h5py
import numpy as np
import torch
from tqdm import tqdm

from gym_drldoe.envs.simple_gp_envs import RandomGridGPEnv
from rlkit.policies.argmax import ArgmaxDiscretePolicy


def load_policy(args):
    loaded = torch.load(args.rl_policy_path)
    return ArgmaxDiscretePolicy(loaded['trainer/qf'].cpu())

def evaluate(args):
    if args.policy_type == 'rl':
        policy = load_policy(args)
    elif args.policy_type == 'ucb':
        raise NotImplementedError('TODO')
        # policy = ucb_policy
    env = RandomGridGPEnv(joint_info=args.joint_info)
    regrets = []
    simple_regrets = []
    for seed in tqdm(range(args.num_evals)):
        regret = []
        simp = []
        state = env.reset(seed=seed)
        for _ in range(args.episode_length):
            act = policy.get_action(state)[0]
            state, reg, _, _ = env.step(act)
            reg *= -1
            regret.append(reg)
            if len(simp) == 0:
                simp.append(reg)
            else:
                simp.append(min(simp[-1], reg))
        regrets.append(np.array(regret))
        simple_regrets.append(np.array(simp))
    regrets = np.array(regrets)
    simple_regrets = np.array(simple_regrets)
    with h5py.File(args.save_path, 'w') as hdata:
        hdata.create_dataset('regrets', data=regrets)
        hdata.create_dataset('simple_regrets', data=simple_regrets)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--policy_type', required=True, choices=['rl', 'ucb'])
    parser.add_argument('--rl_policy_path')
    parser.add_argument('--joint_info', action='store_true')
    parser.add_argument('--episode_length', type=int, default=50)
    parser.add_argument('--num_evals', type=int, default=100)
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    evaluate(args)
