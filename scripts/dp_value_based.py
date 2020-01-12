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
from multiprocessing import Pool

import argparse
import rclpy


class DpValueBased:

    def __init__(self):
        self.action_space = 9
        self.Q = np.zeros((50, 50, 36, self.action_space))
        self.V = np.zeros((500, 500, 1))
        self.time_interval = 0.1
        self.person_velocity = (0.5, 0.1)
        self.use_v = True
        self.depth = 4  # td depth lambda
        self.gamma = 0.8
        self.alpha = 0.005
        self.max_num_episodes = 10000000
        self.save_lock = threading.Lock()
        self.initialize_v_with_reward()
        number_of_run = 500
        # for i in range (number_of_run):
        #     start = time.time()
        #     self.update_v()
        #     print("time to finish: min {} hours {} i: {}".format(((number_of_run-i) * (time.time()-start))/60, ((number_of_run-i) * (time.time()-start))/60/60, i))
        #     self.save_threaded()
        self.visualize_v()
        #self.load_q()
        # self.visualize()

    """
    va, vb: linear vel person, robot
    a, b: angular vel person, robot
    x1, x2, x3: x, y, angle person-robot
    h: time
    
    return: new x1, x2, x3
    """
    @staticmethod
    def model(va, vb, a, b, x1, x2, x3, h):
        x3 = DpValueBased.zero_2pi(x3)
        a = DpValueBased.pi_pi(a)
        b = DpValueBased.pi_pi(b)
        # print ("va {} vb {} a {} b {} x1 {} x2 {} x3 {} h {}".format(va, vb, a, b, x1, x2, x3, h))
        x1_n = x1 + h * (-va + vb * math.cos(x3) + a * x2)
        x2_n = x2 + h * (vb * math.sin(x3) - a * x1)
        x3_n = DpValueBased.zero_2pi(x3 + h * b - a)
        return x1_n, x2_n, x3_n

    def visualize_v(self):
        V = self.V - np.min(self.V)
        norm_multiplier = np.max(V)
        for i in range(self.V.shape[2]):
            image = V[:, :, i]  / norm_multiplier
            print(i / self.Q.shape[2] * 360)
            im = np.array(image * 255, dtype = np.uint8)
            threshed = cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 0)
            mask_gray = cv.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

            image = cv.applyColorMap(mask_gray, cv.COLORMAP_BONE	)
            image = cv.resize(image, (image.shape[0]*2, image.shape[1]*2))

            cv.imshow("image", image)
            cv.waitKey()

    def visualize_q(self):
        q_a = self.Q.argmax(-1)
        for i in range (self.Q.shape[2]):
            image = q_a[:,:,i] / 9.
            print (i/self.Q.shape[2] *360)
            cv.imshow("image", image)
            cv.waitKey()

    def save_threaded(self, vq="v"):
        thread1 = threading.Thread(target=self.save_vq, args=(vq))
        thread1.start()

    def save_vq(self, name):
        with self.save_lock:
            print ("start saving")
            if name == "v":
                np.save("v", self.V)
            elif name == "q":
                np.save("q", self.Q)
            else:
                print ("error only v or q can be saved")
            print ("saved {}".format(name))

    def save_q(self):
        with self.save_lock:
            print ("start saving")
            np.save("test", self.Q)
            print ("saved")

    def load_q(self):
        if os.path.exists("q.npy"):
            print ("loading q")
            self.Q = np.load("q.npy")

    def get_best_action(self, heading, pos, current_velocity):
        x_index, y_index, degree_index = self.get_index_from_heading_pos(heading, pos)
        if self.use_v:
            reward, action = self.depth_recursive_reward(pos[0], pos[1], heading, self.depth, current_velocity, get_action=True, use_random_speed_person=False)
        else:
            actions = self.Q[x_index, y_index, degree_index]
        return action

    def get_robot_state_from_table(self,x, y, a):
        heading = a * 2 * math.pi / self.V.shape[2]
        x1 = (x - self.V.shape[0]/2)/(self.V.shape[0]/2/5)
        x2 = (y - self.V.shape[1]/2)/(self.V.shape[1]/2/5)

        return x1, x2, heading

    def get_index_from_heading_pos(self, heading, pos):
        heading = DpValueBased.zero_2pi(heading)
        degree = np.rad2deg(heading)
        degree_index = math.floor(degree * self.V.shape[2] / 360.)
        x = min(max(pos[0], -5), 5) + 5
        x_index = math.floor(x * self.V.shape[0]/2/5)
        y = min(max(pos[1], -5), 5) + 5
        y_index = math.floor(y *  self.V.shape[0]/2/5)

        x_index = min(max(x_index, 0), self.V.shape[0]-1)
        y_index = min(max(y_index, 0), self.V.shape[1]-1)
        degree_index_before = degree_index
        degree_index = min(max(degree_index, 0), self.V.shape[2]-1)
        if abs(degree_index - degree_index_before) > 1:
            print ("degree index may be wront: prev, now: {} {}".format(degree_index_before, degree_index))
        return (x_index, y_index, degree_index)

    def set_q_value(self, q_value, heading, pos, action):
        x_index, y_index, degree_index = self.get_index_from_heading_pos(heading, pos)
        self.Q[x_index, y_index, degree_index, action] = q_value

    def get_q_value(self, heading, pos, action):
        x_index, y_index, degree_index = self.get_index_from_heading_pos(heading, pos)

        return self.Q[x_index, y_index, degree_index, action]

    def action_angular_linear(self, action, current_velocity=(None,None)):
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
        if current_velocity[0] != None:
            angular = (angular + current_velocity[1])/2.
            linear = (linear + current_velocity[0])/2.
        return angular, linear

    def is_collided (self, state):
        pos_rel = state[0:2]
        distance = math.hypot(pos_rel[0], pos_rel[1])
        if distance < 0.3:
            return True
        else:
            return False

    def calculate_reward(self, state):
        angle_robot_person = np.rad2deg(DpValueBased.pi_pi(state[2]))
        pos_rel = state[0:2]
        distance = math.hypot(pos_rel[0], pos_rel[1])
        angle_robot_person = math.atan2(pos_rel[1], pos_rel[0])
        angle_robot_person = np.rad2deg(angle_robot_person)
        # Negative reward for being behind the person
        reward = 0
        if distance<0.3:
            reward = -1.3
        elif abs(distance - 1.7) < 0.7:
            reward += 0.1 * (0.7 - abs(distance - 1.7))
        elif distance >= 1.7:
            reward -= 0.16 * (distance-1.7)
        elif distance < 1:
            reward -= (1 - distance) * 2
        if abs(angle_robot_person) < 45:
            reward += 0.20 * (45 - abs(angle_robot_person)) / 45
        else:
            reward -= 0.80 * (abs(angle_robot_person) - 45) / 180
        # if not 90 > angle_robot_person > 0:
        #     reward -= distance/6.0
        # elif self.min_distance < distance < self.max_distance:
        #     reward += 0.1 + (90 - angle_robot_person) * 0.9 / 90
        # elif distance < self.min_distance:
        #     reward -= 1 - distance / self.min_distance
        # else:
        #     reward -= distance / 7.0
        reward = min(max(reward, -1), 1)
        return reward

    def initialize_v_with_reward(self):
        if os.path.exists("v.npy"):
            self.V = np.load("v.npy")
            print("init v using v.npy")
            return
        for x in range (self.V.shape[0]):
            for y in range (self.V.shape[1]):
                for a in range (self.V.shape[2]):
                    x1, x2, heading = self.get_robot_state_from_table(x, y, a)
                    reward = self.calculate_reward((x1, x2, heading))
                    self.V[x, y, a] = reward

    def depth_recursive_reward(self, x, y, a, depth, current_velocity, get_action=False, use_random_speed_person=False):
        if depth <= 0:
            xi_1, xi_2, xi_3  = self.get_index_from_heading_pos(a, (x,y))
            return self.V[xi_1, xi_2, xi_3]
        if self.is_collided((x,y,a)):
            return -1
        # xi_1, xi_2, xi_3  = self.get_index_from_heading_pos(a, (x,y))
        state_reward = self.calculate_reward((x, y, a))
        reward_list = []
        for action in range(9):
            angular, linear = self.action_angular_linear(action, current_velocity)
            current_velocity = (linear, angular)
            if use_random_speed_person:
                person_angular = random.random() - 0.5
                person_linear = random.random() - 0.5
            else:
                person_angular = self.person_velocity[1]
                person_linear = self.person_velocity[0]
            x1, x2, x3 = self.model(person_linear, linear, person_angular, angular, x, y, a, self.time_interval)
            reward_r = self.depth_recursive_reward(x1, x2, x3, depth-1, current_velocity, use_random_speed_person=use_random_speed_person)
            reward = state_reward + self.gamma * reward_r
            reward_list.append(reward)
        if get_action:
            print ("best {}, list: {}".format(np.argmax(reward_list),reward_list))
            return max(reward_list), np.argmax(reward_list)
        return max(reward_list)

    def update_v_thread(self, x, y, a_start, a_end, current_velocity):
        for a in range(a_start, a_end):
            x1, x2, heading = self.get_robot_state_from_table(x, y, a)
            reward = self.depth_recursive_reward(x1, x2, heading, self.depth, current_velocity, use_random_speed_person=False)
            with self.save_lock:
                self.V[x, y, a] = self.V[x, y, a] + self.alpha * (reward - self.V[x, y, a])

    def update_v(self):
        for x in range (self.V.shape[0]):
            for y in range (self.V.shape[1]):
                len_a = self.V.shape[2]
                num_thread = 18
                current_velocity_robot = (0,0)
                threads = [threading.Thread(target=self.update_v_thread, args=(x, y, i*math.ceil(36./num_thread), min((i+1)*math.ceil(36./num_thread), 36), current_velocity_robot)) for i in range(num_thread)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()


            print ("x {}".format(x, y))

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

    @staticmethod
    def zero_2pi(angle):
        while angle > 2 * math.pi:
            angle -= 2 * math.pi
        while angle < 0:
            angle += 2 * math.pi
        return angle

    @staticmethod
    def pi_pi(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

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
                if random.random() > 1:
                    action = random.randint(0, self.action_space-1)
                else:
                    action = self.get_best_action(heading, pose, robot_velocity)

                angular, linear = self.action_angular_linear(action)
                # angular = -0.5
                # linear = 0.5
                observation, reward, over, _ = env.step((linear, angular))
                x = DpValueBased.model(person_velocity[0], robot_velocity[0], person_velocity[1], robot_velocity[1], pose[0], pose[1], heading, 0.077)

                print("estimated vs real: x1:{} {} x2:{} {} x3:{} {}".format(x[0], pose[0], x[1], pose[1], x[2], DpValueBased.zero_2pi(heading)))
                heading, pose, person_velocity, robot_velocity = observation
                self.person_velocity = person_velocity
                obs_reward.append((observation, reward, over, action))
                rewards.append(reward)
                # if len(obs_reward) > self.depth:
                #     self.update_q(obs_reward, len(obs_reward)-self.depth,self.depth)
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
            # for x in range (1, min(self.depth, len(obs_reward))):
            #     self.update_q(obs_reward, len(obs_reward) - x, x)
            # if num_episodes % 100 == 0:
            #     if self.use_v:
            #         self.save_threaded("v")
            #     else:
            #         self.save_threaded("q")
            #wandb.log({"reward": np.average(rewards)})
            env.reset()

if __name__ == '__main__':
    #wandb.init(project="followahead_dp")
    dp = DpValueBased()
    dp.train()
    # wandb.init(project="followahead_rldp")
    parser = argparse.ArgumentParser(description='input weight file of the network')


