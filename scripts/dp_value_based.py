import wandb
import time
import random
import gym
import os
import gym_gazebo
import cv2 as cv
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import threading
import math

import argparse
import rclpy


class DpValueBased:

    def __init__(self):
        self.action_space = 9
        self.Q = np.zeros((100, 100, 72, self.action_space))

        self.load_q()
        # self.visualize()
        self.save_lock = threading.Lock()
        self.depth = 12 # td depth lambda
        self.gamma = 0.8
        self.alpha = 0.0001
        self.max_num_episodes = 10000000

    """
    va, vb: linear vel person, robot
    a, b: angular vel person, robot
    x1, x2, x3: x, y, angle person-robot
    h: time
    
    return: new x1, x2, x3
    """
    @staticmethod
    def model(va, vb, a, b, x1, x2, x3, h):
        x1_n = x1 + h * (-va + vb * math.cos(x3) + a * x2)
        x2_n = x2 + h * (-vb * math.sin(x3) - a * x1)
        x3_n = x3 + h * (b - a)
        return x1_n, x2_n, x3_n

    def visualize(self):
        q_a = self.Q.argmax(-1)
        for i in range (self.Q.shape[2]):
            image = q_a[:,:,i] / 9.
            print (i/self.Q.shape[2] *360)
            cv.imshow("image", image)
            cv.waitKey()

    def save_threaded(self):
        thread1 = threading.Thread(target=self.save_q, args=())
        thread1.start()

    def save_q(self):
        with self.save_lock:
            print ("start saving")
            np.save("test", self.Q)
            print ("saved")

    def load_q(self):
        if os.path.exists("test.npy"):
            print ("loading q")
            self.Q = np.load("test.npy")

    def get_best_action(self, heading, pos):
        x_index, y_index, degree_index = self.get_index_from_heading_pos(heading, pos)
        actions = self.Q[x_index, y_index, degree_index]
        print ("actions: {} amax: {} heading: {} pos: {}".format(actions, actions.argmax(), heading, pos))
        return actions.argmax()

    def get_index_from_heading_pos(self, heading, pos):
        degree = np.rad2deg(heading)
        degree = 360 + degree if degree < 0 else degree
        degree_index = int(degree * 72 / 360.)
        x = min(max(pos[0], -5), 5) + 5
        x_index = int(x * 10)
        y = min(max(pos[1], -5), 5) + 5
        y_index = int(y * 10)

        x_index = min(max(x_index, 0), 99)
        y_index = min(max(y_index, 0), 99)
        degree_index = min(max(degree_index, 0), 71)
        return (x_index, y_index, degree_index)

    def set_q_value(self, q_value, heading, pos, action):
        x_index, y_index, degree_index = self.get_index_from_heading_pos(heading, pos)
        self.Q[x_index, y_index, degree_index, action] = q_value

    def get_q_value(self, heading, pos, action):
        x_index, y_index, degree_index = self.get_index_from_heading_pos(heading, pos)

        return self.Q[x_index, y_index, degree_index, action]

    def action_angular_linear(self, action):
        angular = 0
        linear = 0
        if action == 1:
            linear = 0.5
        elif action == 2:
            linear = 1
        elif action == 3:
            angular = 0.5
        elif action == 4:
            angular = -0.5
        elif action == 5:
            angular = 0.5
            linear = 0.5
        elif action == 6:
            linear = 0.5
            angular = -0.5
        elif action == 7:
            angular = 0.5
            linear = 1
        elif action ==8:
            angular = -0.5
            linear = 1

        return angular, linear

    def update_q(self, obs_reward, index, depth):
        print ("update index {}, depth {}, len_obs {}".format(index, depth, len(obs_reward)))
        observation_q, reward, over, action_q = obs_reward[index]
        q = self.get_q_value(observation_q[0], observation_q[1], action_q)
        reward_q = reward
        counter = 1
        if depth == 1:
            q_last = 0
        else:
            for observation, reward, over, action in obs_reward[index + 1: index+depth]:
                reward_q += np.power(self.gamma, counter) * reward
                counter += 1
            q_last = self.get_q_value(observation[0], observation[1], action)
        reward_q += np.power(self.gamma, self.depth) * q_last
        self.set_q_value(q + self.alpha * (reward_q - q), observation_q[0], observation_q[1], action_q)

    def train(self):
        num_episodes = 0
        env = gym.make('gazebo-v0')
        env.set_agent(0)
        env.resume_simulator()
        heading, pose, person_velocity, robot_velocity = env.get_observation()
        while num_episodes < self.max_num_episodes:
            over = False
            env.resume_simulator()
            rewards = []
            obs_reward = []
            num_episodes += 1
            while not over:
                if random.random() > 0.5:
                    action = random.randint(0, self.action_space-1)
                else:
                    action = self.get_best_action(heading, pose)

                angular, linear = self.action_angular_linear(action)
                print("before step")
                observation, reward, over, _ = env.step((linear, angular))
                x = DpValueBased.model(person_velocity[0], robot_velocity[0], person_velocity[1], robot_velocity[1], pose[0], pose[1], heading, 0.05*2.42)
                heading, pose, person_velocity, robot_velocity = observation
                print("estimated vs real: x1:{} {} x2:{} {} x3:{} {}".format(x[0], pose[0], x[1], pose[0], x[2], pose[1]))

                obs_reward.append((observation, reward, over, action))
                rewards.append(reward)
                if len(obs_reward) > self.depth:
                    self.update_q(obs_reward, len(obs_reward)-self.depth,self.depth)
                    # observation_q, reward, over, action_q = obs_reward[-self.depth]
                    # q = self.get_q_value(observation_q[0], observation_q[1], action_q)
                    # reward_q = reward
                    # counter = 1
                    # for observation, reward, over, action in obs_reward[-self.depth+1:]:
                    #     reward_q += np.power(self.gamma, counter)*reward
                    #     counter += 1
                    # q_last = self.get_q_value(observation[0], observation[1], action)
                    # reward_q += np.power(self.gamma, self.depth) * q_last
                    # self.set_q_value(q + self.alpha*(reward_q - q), observation_q[0], observation_q[1], action_q)
            for x in range (1, min(self.depth, len(obs_reward))):
                self.update_q(obs_reward, len(obs_reward) - x, x)
            if num_episodes % 100 == 0:
                self.save_threaded()
            #wandb.log({"reward": np.average(rewards)})
            print("before reset {}".format(num_episodes))
            env.reset()

if __name__ == '__main__':
    #wandb.init(project="followahead_dp")
    dp = DpValueBased()
    dp.train()
    # wandb.init(project="followahead_rldp")
    parser = argparse.ArgumentParser(description='input weight file of the network')


