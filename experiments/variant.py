variant = dict(
    mlflow_uri="http://128.2.210.74:8080",
    gpu=True,    
    algorithm="SAC",
    version="normal",
    layer_size=256,
    layer_size_actor=256,
    UCB_rate=0.2,
    gamma_a=30,
    gamma_b=1.5,     
    num_gammas=10,
    random_initial=False,
    replay_buffer_size=int(1E6),
    algorithm_kwargs=dict(
        num_epochs=500,
        num_eval_steps_per_epoch=4000,
        num_trains_per_train_loop=1000,         # May need to reduce to avoid overtraining
        num_expl_steps_per_train_loop=10000,
        min_num_steps_before_training=10000,
        max_path_length=1000,
        batch_size=256,
    ),
    trainer_kwargs=dict(
        discount=0.99,                          # Need to tune
        soft_target_tau=5e-3,
        target_update_period=1,
        policy_lr=3E-4,                         # Need to tune
        qf_lr=3E-4,                             # Set same as policy
        reward_scale=5,                         # Need to tune
        use_automatic_entropy_tuning=True,
        # target_entropy= -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
    ),
)

env_variant = dict(
    env_str='HalfCheetah-v2',   
    # num_test_points=50,
    # region_length=20,
    # burnin=0,
    # reciprocal=True
)