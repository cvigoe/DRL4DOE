from gym.envs.registration import register

register(
    id='drl4doe-v0',
    entry_point='gym_drl4doe.envs:DRL4DOE',
)