import wandb
import time
import random
import gym
import os
import gym_gazebo
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import threading

import argparse
import rclpy


class DpValueBased:

    def __init__(self):
        self.action_space = 9
        self.Q = np.zeros((100, 100, 72, self.action_space))
        self.load_q()
        self.save_lock = threading.Lock()
        self.depth = 12 # td depth lambda
        self.gamma = 0.8
        self.alpha = 0.0001
        self.max_num_episodes = 10000000

    def save_threaded(self):
        thread1 = threading.Thread(target=self.save_q, args=())
        thread1.start()

    def save_q(self):
        with self.save_lock:
            print ("start saving")
            np.save("test", self.Q)
            print ("saved")

    def load_q(self):
        print ("loading q")
        self.Q = np.load("test.npy")

    def get_best_action(self, heading, pos):
        x_index, y_index, degree_index = self.get_index_from_heading_pos(heading, pos)
        actions = self.Q[x_index, y_index, degree_index]
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
            obs_reward = []
            num_episodes += 1
            while not over:
                if random.random() > 0.3:
                    action = random.randint(0, self.action_space-1)
                else:
                    action = self.get_best_action(heading, pose)

                angular, linear = self.action_angular_linear(action)
                print("before step")
                observation, reward, over, _ = env.step((linear, angular))
                obs_reward.append((observation, reward, over, action))
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
            print("before reset {}".format(num_episodes))
            env.reset()

if __name__ == '__main__':
    dp = DpValueBased()
    dp.train()
    # wandb.init(project="followahead_rldp")
    parser = argparse.ArgumentParser(description='input weight file of the network')


