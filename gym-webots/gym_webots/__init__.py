from gym.envs.registration import register

register(
    id='webots-v0',
    entry_point='gym_webots.envs.gym_webots:WebotsEnv',
)