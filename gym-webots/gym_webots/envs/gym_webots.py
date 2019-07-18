from gym.utils import seeding

import os, subprocess, time, signal

import gym

import math

import rospy
from webots_ros.srv import get_bool
from webots_ros.srv import set_int
from webots_ros.srv import get_int
from webots_ros.srv import set_intRequest

import threading

import logging

logger = logging.getLogger(__name__)


class Service:
    def __init__(self, str, type_off):
        self.str = str
        self.type_off = type_off
        self.check()

    def check(self):
        rospy.wait_for_service(self.str)
        self.srv = rospy.ServiceProxy(self.str, self.type_off)

    def call(self, value):
        self.srv.call(value)

class WebotsEnv(gym.Env):

    def __init__(self):
        self.node = rospy.init_node("webots_env", anonymous=True)
        self.manager_service_str = '/manager/supervisor/'
        self.robot_service_str = '/pioneer3at/'

        self.services = {}
        self.services['reset'] = Service(self.manager_service_str + 'simulation_reset', get_bool)
        self.services['get_state'] = Service(self.manager_service_str + 'simulation_get_mode', get_int)
        self.services['reset'].call(True)
        while True:
            try:
                time.sleep(0.1)
                self.services['get_state'].call(True)
                break
            except Exception as e:
                rospy.loginfo_throttle(1, "still waiting for reset to finish")

        trying = ['gps/enable', 'inertial_unit/enable', 'accelerometer/enable', 'gyro/enable', 'Sick_LMS_291/enable']
        while len(trying) > 0:
            failed = []
            for name in trying:
                self.services[name.split('/')[0]] = Service(self.robot_service_str + name, set_int)
                try:
                    self.services[name.split('/')[0]].call(1)
                    rospy.loginfo(name + ' called')
                except Exception as e:
                    rospy.loginfo(e,name+' called')
                    failed.append(name)
                    continue
            trying = failed
        x.call(True)

        self.reset_service_str = self.manager_service_str +'simulation_reset'
        self.gps_service_str = self.robot_service_str + 'gps/enable'

        rospy.wait_for_service(self.reset_service_str)
        self.reset_srv = rospy.ServiceProxy(self.reset_service_str, get_bool)
        self.reset_srv.call(True)

        self.observation_space = gym.spaces.Tuple(
                (
                    gym.spaces.Box(low=0, high=1, shape=(100, 100)),
                    gym.spaces.Box(low=0, high=1, shape=(8,))
                )
            )
        # Action space omits the Tackle/Catch actions, which are useful
        # on defense
        self.action_space = gym.spaces.Discrete(self.env.get_action_space())
        self.min_distance = 1
        self.max_distance = 3.5
        self.number_of_steps = 0
        self.max_numb_steps = 40
        self.distance = 0
        self.reward_range = [-1, 1]
        self._person_rand_thread = threading.Thread(
            target=self.person_random_point, args=(), daemon=True)
        self._person_rand_thread.start()



    def person_random_point(self):
        while (True):
            time.sleep(.1)

    def __del__(self):
        # todo
        return

    def take_action(self, action):
        # todo
        return
    def step(self, action):
        self.number_of_steps += 1
        self.env.take_action(action)
        reward = self.get_reward()
        ob = self.env.get_observation()
        episode_over = False
        rel_person = self.env.get_person_position_relative_robot()
        self.distance = math.hypot(rel_person[0], rel_person[1])
        if self.env.is_collided():
            episode_over = True
            print('collision happened episode over')
            reward -= 1
        elif self.distance > 5:
            episode_over = True
            print('max distance happened episode over')
        elif self.number_of_steps > self.max_numb_steps:
            episode_over = True
            print('max number of steps episode over')

        return ob, reward, episode_over, {}

    def get_reward(self):
        pos_rel = self.env.get_person_position_relative_robot()
        reward = 0
        distance = math.hypot(pos_rel[0], pos_rel[1])
        angle_robot_person = self.env.get_angle_person_robot()
        print('angle:', angle_robot_person)
        # Negative reward for being behind the person
        if self.env.is_collided():
            reward -= 1
        if not 90 > angle_robot_person > 0:
            reward -= distance/6.0
        elif self.min_distance < distance < self.max_distance:
            reward += 0.1 + (90 - angle_robot_person) * 0.9 / 90
        elif distance < self.min_distance:
            reward -= 1.0/distance
        else:
            reward -= distance / 7.0
        reward = min(max(reward, -1), 1)
        # ToDO check for obstacle
        print(reward)
        return reward

    def reset(self):

        self.env.init_simulator()
        self.number_of_steps = 0
        """ Repeats NO-OP action until a new episode begins. """

        return self.env.get_observation()

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        return
