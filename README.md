# 🤖 DRL4DOE

## Installation
Run `pip install -e .` in the root directory.

## Usage
In the main RL script include:

    import gym
    import gym_drl4doe
    ...
    env_variant = ...
    ...
    env = gym.make(env_variant['env_str'])
    env.initialise_environment(**env_variant)

Then use `env` like any normal gym enviornemnt (e.g. use `env.reset()` and `env.step(action)`).
