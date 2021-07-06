from gym.envs.registration import register

register(
    id='drl4doe-v0',
    entry_point='gym_drldoe.envs.drl4doe_env:DRL4DOE',
)
