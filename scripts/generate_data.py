import numpy as np
import cv2 as cv
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import gym
import gym_webots
import os

import pickle

import time

def visualize_data():
    pickle_file_write_idx = 0
    for x in range(2000):
        with open('data/dataset/0/{}.pkl'.format(x), 'rb') as f:
            pickle_file = pickle.load(f)
            image = np.zeros((500,500,3))
            x, y, heading_angle = pickle_file[0][1][0:3]
            pt1 = (x * 50 + 250, y * 50 + 250)
            pt2 = (pt1[0] + math.cos(heading_angle) * 30, pt1[1] + math.sin(heading_angle)*30)

            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))

            pt_goal = pickle_file[1]
            pt_goal = (int(pt_goal[0]  * 50 + 250),  int(pt_goal[1] * 50 + 250))
            cv.arrowedLine(image, pt1, pt2, (0,0,255), 5, tipLength=0.6)
            cv.circle(image, pt_goal,20, (255,255,0))
            # cv.imshow("d", pickle_file[0][0])
            cv.imshow("d", image)
            cv.waitKey(0)

def change_prevdata():
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
    visualize_data()
    exit(0)
    data = []
    pkl_counter = 2874
    env = gym.make('webots-v0')
    env.set_robot_to_auto()
    env.resume_simulator()
    prev_obs = env.get_observation()
    time.sleep(1)
    prev_goal, prev_angle = env.get_goal_person()
    while(True):

        next_obs = env.get_observation()
        pos_goal, angle_distance = env.get_goal_person()
        if pos_goal is None or angle_distance is None or (prev_angle == angle_distance and prev_goal == pos_goal) :
            continue
        data = (next_obs, np.asarray(angle_distance))
        with open('data/dataset/0/{}.pkl'.format(pkl_counter), 'wb') as f:
            pickle.dump(data, f)
            data = []
            print ("saving {}.pkl".format(pkl_counter))
            pkl_counter+=1
        prev_obs = next_obs
        prev_goal, prev_angle = (pos_goal, angle_distance)

        time.sleep(0.05)