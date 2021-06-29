
"""
Run DQN on grid world.
"""
import argparse

import gym
from torch import nn as nn

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from gym_drldoe.envs.simple_gp_envs import RandomGridGPEnv


def s2i(string):
    """Make a comma separated string of ints into a list of ints."""
    if ',' not in string:
        if len(string) > 0:
            return [int(string)]
        else:
            return []
    return [int(s) for s in string.split(',')]

def experiment(variant):
    expl_env = RandomGridGPEnv(**variant['env_kwargs'])
    eval_env = RandomGridGPEnv(**variant['env_kwargs'])
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=obs_dim,
        output_size=action_dim,
    )
    target_qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=obs_dim,
        output_size=action_dim,
    )
    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space),
        eval_policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant['trainer_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--q_architecture', default='500,500')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--joint_info', action='store_true')
    parser.add_argument('--cuda_device', default='')
    args = parser.parse_args()
    variant = dict(
        algorithm="DDQN",
        version="normal",
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=args.epochs,
            num_eval_steps_per_epoch=500,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=500,
            max_path_length=50,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=3E-4,
        ),
        policy_kwargs=dict(
            q_architecture=args.q_architecture,
        ),
        env_kwargs=dict(
            joint_info=args.join_info,
        )
    )
    setup_logger(args.run_id, variant=variant)
    ptu.set_gpu_mode(args.cuda_device != '', gpu_id=args.cuda_device)
    experiment(variant)
