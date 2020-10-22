from gym.envs.registration import register

register(
    id='vicon-v0',
    entry_point='gym_vicon.envs.gym_vicon:ViconEnv',
)
