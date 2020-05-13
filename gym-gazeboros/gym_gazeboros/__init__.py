from gym.envs.registration import register

register(
    id='gazeboros-v0',
    entry_point='gym_gazeboros.envs.gym_gazeboros:GazeborosEnv',
)
