import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import gym
import gym_webots
import os

import pickle

import time


def read_data():
    """
    "    Change the previous data into new standard dataset data
    """
    pickle_file_write_idx = 0
    for x in range(38):
        with open('data/dataset/{}.pkl'.format(x), 'rb') as f:
            pickle_file = pickle.load(f)
            for p in pickle_file:
                if p is None or p[0] is None or p[1] is None or p[0][0] is None or p[0][1] is None or len(p[1].shape)==0:
                    print(p[1])
                    continue
                with open('data/dataset/0/{}.pkl'.format(pickle_file_write_idx), 'wb') as f:
                    pickle_file_write_idx += 1
                    pickle.dump(p, f)

if __name__ == '__main__':
    read_data()
    exit(0)
    data = []
    pkl_counter = 0
    env = gym.make('webots-v0')
    env.set_robot_to_auto()
    env.resume_simulator()
    while(True):

        next_obs = env.get_observation()
        pos_goal, angle_distance = env.get_goal_person()
        data.append((next_obs, np.asarray(angle_distance)))
        if len(data) > 100:
            with open('data/dataset/{}.pkl'.format(pkl_counter), 'wb') as f:
                pickle.dump(data, f)
                data = []
                print ("saving {}.pkl".format(pkl_counter))
                pkl_counter+=1

        time.sleep(0.05)