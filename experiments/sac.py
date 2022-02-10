import gym
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import gym_drldoe

import mlflow
from mlflow.tracking import MlflowClient
import sys
import collections

from variant import *

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def experiment(variant, env_variant):
    expl_env = gym.make(env_variant['env_str'])  
    eval_env = gym.make(env_variant['env_str'])

    expl_env.initialise_environment(**env_variant,ucb=True)
    eval_env.initialise_environment(**env_variant,ucb=False)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
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

    for seed in range(10):
        LR_pow = 2+(np.random.rand()*4)
        LR = 10**(-LR_pow)
        reward_scale = np.random.rand()*5

        variant['trainer_kwargs']['policy_lr'] = LR
        variant['trainer_kwargs']['qf_lr'] = LR
        variant['trainer_kwargs']['reward_scale'] = reward_scale

        experiment_name = sys.argv[1]
        run_name = sys.argv[2] + '_seed_' + str(seed)
        note = sys.argv[3]

        setup_logger(experiment_name, variant=variant)
        if variant['gpu']:
            ptu.set_gpu_mode(True)
        mlflow.set_tracking_uri(variant['mlflow_uri'])
        mlflow.set_experiment(experiment_name)
        client = MlflowClient()  
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(flatten_dict(variant))
            mlflow.log_params(flatten_dict(env_variant))
            client.set_tag(run.info.run_id, "mlflow.note.content", note)
            experiment(variant, env_variant)