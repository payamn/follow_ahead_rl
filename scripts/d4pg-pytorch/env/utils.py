import gym
#from utils.utils import NormalizedActions

from .pendulum import PendulumWrapper
from .bipedal import BipedalWalker
from .lunar_lander_continous import LunarLanderContinous
from .gazebo_continous import GazeboContinous

def create_env_wrapper(config):
    env = config['env'].lower()
    if env == "pendulum-v0":
        return PendulumWrapper(config)
    elif env == "bipedalwalker-v2":
        return BipedalWalker(config)
    elif env == "lunarlandercontinuous-v2":
        return LunarLanderContinous(config)
    elif env == "gazebo-v0":
        return GazeboContinous(config)
    else:
        raise ValueError("Unknown environment.")