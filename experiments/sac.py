import gym
import numpy as np
import multiprocessing

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import (MdpPathCollector, 
    explMdpPathCollector, gammaExplMdpPathCollector, gammaEvalMdpPathCollector)
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, MakeUCB
from rlkit.torch.sac.sac import SACTrainer, gammaSACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import gym_drldoe
import pudb

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

    if env_variant['env_str'] == 'drl4doe-v0':
        expl_env.initialise_environment(**env_variant)
        eval_env.initialise_environment(**env_variant)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    M_actor = int(variant['layer_size_actor'])
    gamma_a = variant['gamma_a']
    gamma_b = variant['gamma_b']
    num_gammas = variant['num_gammas']
    # UCB_rate = variant['UCB_rate']
    
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + 1,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + 1,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + 1,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + 1,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + 1,
        action_dim=action_dim,
        hidden_sizes=[M_actor, M_actor],
    )
    eval_policy = MakeDeterministic(policy)
    # expl_policy = MakeUCB(policy, UCB_rate=UCB_rate)
    expl_policy = policy
    eval_path_collector = gammaEvalMdpPathCollector(
        eval_env,
        eval_policy,
        gamma=variant['trainer_kwargs']['discount']
    )
    if variant['random_initial']:    
        expl_path_collector = explMdpPathCollector(
            expl_env,
            expl_policy,
        )
    else:
        expl_path_collector = gammaExplMdpPathCollector(
            expl_env,
            expl_policy,
            a=gamma_a,
            b=gamma_b,
        )

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = gammaSACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        num_gammas=num_gammas,
        a=gamma_a,
        b=gamma_b,
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

def mlflow_experiment(experiment_name, run_name, note, variant, seed):
    # LR_coeff = 1+(np.random.rand()*4)
    # LR = LR_coeff*(1e-4)
    # reward_scale = 5+np.random.rand()*10
    # layer_size_actor = np.random.choice([64, 128, 256])
    # discount = 0.75 + np.random.rand()/4
    # UCB_rate = (np.random.rand()/4)
    # NUM_MC_ITERS = int(np.random.rand()*500)

    # variant['trainer_kwargs']['policy_lr'] = LR
    # variant['trainer_kwargs']['qf_lr'] = LR
    # variant['trainer_kwargs']['reward_scale'] = reward_scale
    # variant['trainer_kwargs']['discount'] = discount
    # variant['layer_size_actor'] = float(layer_size_actor)
    # variant['UCB_rate'] = UCB_rate
    # env_variant['NUM_MC_ITERS'] = NUM_MC_ITERS

    setup_logger(experiment_name, variant=variant)
    if variant['gpu']:
        ptu.set_gpu_mode(True)
    mlflow.set_tracking_uri(variant['mlflow_uri'])
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()  
    with mlflow.start_run(run_name=run_name  + '_seed_' + str(seed)) as run:
        mlflow.log_params(flatten_dict(variant))
        mlflow.log_params(flatten_dict(env_variant))
        client.set_tag(run.info.run_id, "mlflow.note.content", note)
        experiment(variant, env_variant)    

if __name__ == "__main__":

    experiment_name = sys.argv[1]
    run_name = sys.argv[2]
    note = sys.argv[3]
    num_seeds = int(sys.argv[4])

    processes = []    

    for seed in range(num_seeds):
        args = (experiment_name, run_name, note, variant, seed)
        process = multiprocessing.Process(target=mlflow_experiment, args=args)
        process.start()
        processes.append(process)  

    for process in processes:
        process.join()          
