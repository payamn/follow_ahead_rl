from models.engine import load_engine
from utils.utils import read_config
import gym_vicon

if __name__ == "__main__":
    CONFIG_PATH = "d4pg-pytorch/configs/follow_rl.yml"
    config = read_config(CONFIG_PATH)

    engine = load_engine(config)
    engine.train()
