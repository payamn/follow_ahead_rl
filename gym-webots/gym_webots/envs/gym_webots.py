from gym.utils import seeding

import os, subprocess, time, signal

import gym

import math
import _thread

import numpy as np
import cv2 as cv

import rospy
from squaternion import quat2euler

from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan

from webots_ros.srv import get_bool
from webots_ros.srv import set_int
from webots_ros.srv import set_float
from webots_ros.srv import get_int
from webots_ros.srv import get_uint64
from webots_ros.srv import node_remove
from webots_ros.srv import node_get_field
from webots_ros.srv import node_get_fieldRequest
from webots_ros.srv import supervisor_get_from_def
from webots_ros.srv import supervisor_get_from_defRequest
from webots_ros.srv import field_import_node_from_string
from webots_ros.srv import field_import_node_from_stringRequest

from simple_pid import PID

import pickle

import threading

import logging

logger = logging.getLogger(__name__)


class Service:
    def __init__(self, str, type_off):
        self.str = str
        self.type_off = type_off
        self.check()

    def check(self):
        rospy.loginfo("check service " + self.str)
        rospy.wait_for_service(self.str)
        self.srv = rospy.ServiceProxy(self.str, self.type_off)

    def call(self, value):
        while (True):
            try:
                return_msg = self.srv.call(value)
                # rospy.loginfo(self.str + ' called')
                break
            except Exception as e:
                print (e)
                time.sleep(0.1)

        return return_msg


class Supervisor:
    def __init__(self):
        self.root_service_str = '/manager/supervisor/'

        self.services = {}
        self.init_services()

    def get_create_robot_str(self, name, model, translation, rotation):
        robot_str = 'DEF ' + name + ' ' + model + ' { translation ' + translation + ' rotation ' + rotation + \
                                        ' controller "ros" \
                                        controllerArgs "--name=' + name + '" extensionSlot [ Camera { translation 0 0.17 -0.22\
                                        width 256 height 128 motionBlur 500 noise 0.02}\
                                        Accelerometer { lookupTable [ -39.24 -39.24 0.005 39.24 39.24 0.005 ]}\
                                        Gyro {lookupTable [-50 -50 0.005 50 50 0.005 ]}\
                                        SickLms291 {translation 0 0.23 -0.136 noise 0.1}\
                                        GPS {}\
                                        InertialUnit {}\
                                        ]}'
        return robot_str

    def init_services(self):
        self.services['create_object'] = Service(self.root_service_str + "field/import_node_from_string",
                                                 field_import_node_from_string)
        self.services['get_root'] = Service(self.root_service_str + "get_root", get_uint64)
        self.services['get_field'] = Service(self.root_service_str + "node/get_field", node_get_field)
        self.services['get_from_deff'] = Service(self.root_service_str + "get_from_def", supervisor_get_from_def)
        self.services['remove_object'] = Service(self.root_service_str + "node/remove", node_remove)
        self.services['reset'] = Service(self.root_service_str + 'simulation_reset', get_bool)
        self.services['get_state'] = Service(self.root_service_str + 'simulation_get_mode', get_int)

        self.root_node = self.services['get_root'].call(0).value
        self.child_field = self.services['get_field'].call(
            node_get_fieldRequest(node=self.root_node, fieldName='children')).field
        while True:
            try:
                time.sleep(0.1)
                self.services['get_state'].call(True)
                break
            except Exception as e:
                rospy.loginfo_throttle(1, "still waiting for reset to finish", e )

    def remove_object(self, defName):
        input = supervisor_get_from_defRequest(defName)
        response = self.services['get_from_deff'].call(input)
        node = -1
        if response is not None and response.node != 0:
            node = response.node
        else:
            rospy.loginfo("already removed or a problem happened for def: " + defName)
        if node != -1:
            while (self.services['remove_object'].call(node).success != 1):
                rospy.logerr("remove object error: " + defName)

    def create_robot(self, name, model, translation, rotation):
        self.remove_object(defName=name)
        robot_str = self.get_create_robot_str(name, model, translation, rotation)
        self.services['create_object'].call(field_import_node_from_stringRequest(field=self.child_field, position=-1, nodeString=robot_str))


    def reset(self):
        self.services['reset'].call(True)
        self.init_services()

class Robot():
    def __init__(self, name, init_pos, angle, supervisor, model='Pioneer3at'):
        self.model = 'Pioneer3at'
        self.name = name
        self.robot_service_str = '/{}/'.format(self.name)
        self.supervisor = supervisor

        self.supervisor.create_robot(self.name, self.model, str(-init_pos[1]) + " 0.178 " + str(-init_pos[0]), "0 1 0 "+str(angle))
        self.wheels_left = ['back_left_wheel', 'front_left_wheel']
        self.wheels_right = ['back_right_wheel', 'front_right_wheel']
        self.services = {}
        self.max_speed = 6.3
        self.max_laser_range = 25.0 # meter
        self.width_laser_image = 100
        self.height_laser_image = 50
        self.pos = (0, 0)
        self.init_services()
        self.angular_pid = PID(0.45, 0, 0.74, setpoint=0)
        self.linear_pid = PID(4, 0, 0.05, setpoint=0)
        self.orientation = angle
        self.pos_sub = rospy.Subscriber(self.robot_service_str+'/gps/values', NavSatFix, self.position_cb)
        self.imu_sub = rospy.Subscriber(self.robot_service_str+'/inertial_unit/roll_pitch_yaw', Imu, self.imu_cb)
        self.laser_sub = rospy.Subscriber(self.robot_service_str+'/Sick_LMS_291/laser_scan/layer0', LaserScan, self.laser_cb)

    def take_action(self, action):
        speed_left = 0
        speed_right = 0
        if action == 0:
            pass # stop robot
        elif action == 1:
            speed_left = speed_right = self.max_speed
        elif action == 2:
            speed_left = speed_right = -self.max_speed
        elif action == 3:
            speed_left = self.max_speed
            speed_right = -self.max_speed
        elif action == 4:
            speed_left = -self.max_speed
            speed_right = self.max_speed
        elif action == 5:
            speed_left = self.max_speed/2
            speed_right = self.max_speed/2
        elif action == 6:
            speed_left = self.max_speed/2
            speed_right = -self.max_speed/2
        elif action == 7:
            speed_left = -self.max_speed/2
            speed_right = self.max_speed/2

        for wheel in self.wheels_left:
            self.services[wheel + "_vel"].call(speed_left)
        for wheel in self.wheels_right:
            self.services[wheel + "_vel"].call(speed_right)

    def stop_robot(self):
        for wheel in self.wheels_left:
            self.services[wheel + "_vel"].call(0)
        for wheel in self.wheels_right:
            self.services[wheel + "_vel"].call(0)

    def go_to_pos(self, pos):
        angle = math.atan2(pos[1]-self.pos[1], pos[0]-self.pos[0])
        distance = math.hypot(pos[0]-self.pos[0], pos[1]-self.pos[1])
        # diff_angle_prev = (angle - self.orientation + math.pi) % (math.pi * 2) - math.pi
        while not distance < 1.5:
            angle = math.atan2(pos[1] - self.pos[1], pos[0] - self.pos[0])
            distance = math.hypot(pos[0] - self.pos[0], pos[1] - self.pos[1])
            diff_angle = (angle -self.orientation + math.pi) % (math.pi*2) - math.pi
            # if abs(diff_angle_prev - diff_angle) > math.pi*3/:
            #     diff_angle = diff_angle_prev
            # else:
            #     diff_angle_prev = diff_angle
            # angular_vel = min(max(self.angular_pid(math.atan2(math.sin(angle-self.orientation), math.cos(angle-self.orientation)))*200, -self.max_speed/3),self.max_speed)
            angular_vel = min(max(self.angular_pid(diff_angle)*3, -self.max_speed/2),self.max_speed/2)
            linear_vel = min(max(self.linear_pid(-distance), -self.max_speed), self.max_speed)
            # set speed left wheels
            left_vel = linear_vel + angular_vel
            right_vel = linear_vel - angular_vel
            check_speed = max(abs(left_vel), abs(right_vel))
            if check_speed > self.max_speed:
                left_vel = left_vel * self.max_speed / check_speed
                right_vel = right_vel * self.max_speed / check_speed

            for wheel in self.wheels_left:
                self.services[wheel+"_vel"].call(left_vel)
            for wheel in self.wheels_right:
                self.services[wheel+"_vel"].call(right_vel)
            # rospy.loginfo("angle: {} distance: {} vel: angular: {} linear: {} left: {} right: {} diff: {} orientation: {}"\
            #     .format(np.rad2deg(angle), distance, angular_vel, linear_vel, left_vel, right_vel, np.rad2deg(diff_angle), np.rad2deg(self.orientation)))
            time.sleep(0.1)

    def get_pos(self):
        return self.pos

    def imu_cb(self, imo_msg):
        self.orientation = quat2euler(
            imo_msg.orientation.x, imo_msg.orientation.y, imo_msg.orientation.z, imo_msg.orientation.w)[0]

    def laser_cb(self, laser_msg):
        angle_increments = np.arange(float(laser_msg.angle_min) - float(laser_msg.angle_min) , float(laser_msg.angle_max) - 0.001 - float(laser_msg.angle_min), float(laser_msg.angle_increment))
        ranges = np.asarray(laser_msg.ranges, dtype=float)
        remove_index = np.append(np.argwhere(ranges >= self.max_laser_range), np.argwhere(ranges <= laser_msg.range_min))
        angle_increments = np.delete(angle_increments, remove_index)
        ranges = np.delete(ranges, remove_index)

        x = np.floor((self.width_laser_image/2.0 - (self.height_laser_image / self.max_laser_range) * np.multiply(np.cos(angle_increments), ranges))).astype(np.int)
        y = np.floor((self.height_laser_image-1 - (self.height_laser_image / self.max_laser_range) * np.multiply(np.sin(angle_increments), ranges))).astype(np.int)
        if len(x) > 0 and (np.max(x) >= self.width_laser_image or np.max(y) >= self.height_laser_image):
            print ("problem, max x:{} max y:{}".format(np.max(x), np.max(y)))
        scan_image = np.zeros((self.height_laser_image, self.width_laser_image), dtype=np.uint8)
        scan_image[y, x] = 255
        self.scan_image = scan_image
        if "robot" in self.name:
            cv.namedWindow("laser {}".format(self.name), cv.WINDOW_AUTOSIZE)
            cv.imshow("laser {}".format(self.name), scan_image)
            cv.waitKey(1)

    def position_cb(self, pos_msg):
        self.pos = (-pos_msg.longitude, -pos_msg.latitude)
        # rospy.loginfo_throttle(0.2, "current pos: {}, orientation {}".format(self.pos, self.orientation))

    def init_services(self):
        self.services = {}

        trying = ['gps/enable', 'inertial_unit/enable', 'accelerometer/enable', 'gyro/enable', 'Sick_LMS_291/enable']
        while len(trying) > 0:
            failed = []
            for name in trying:
                self.services[name.split('/')[0]] = Service(self.robot_service_str + name, set_int)
                try:
                    self.services[name.split('/')[0]].call(1)
                except Exception as e:
                    rospy.loginfo(e,name+' called')
                    failed.append(name)
                    continue
            trying = failed

        for wheel in self.wheels_left+self.wheels_right:
            # init motors; position to INFINITY to be able to use velocity
            self.services[wheel+"_pos"] = Service(self.robot_service_str + wheel + '/set_position', set_float)
            self.services[wheel + "_pos"].call(float('inf'))
            # init velocity services for motors
            self.services[wheel+"_vel"] = Service(self.robot_service_str + wheel + '/set_velocity', set_float)
            self.services[wheel + "_vel"].call(0)
    def __del__(self):
        rospy.loginfo("removing robot:" + self.name)
        self.supervisor.remove_object(self.name)

class WebotsEnv(gym.Env):

    def calculate_angle_using_path(self, idx):
        return math.atan2(self.path[idx+1][1] - self.path[idx][1], self.path[idx+1][0] - self.path[idx][0])

    def get_person_position_relative_robot(self):
        return np.asarray(self.person.pos) - np.asarray(self.robot.pos)

    def path_follower(self, robot):
        if len(self.path)==0:
            return
        for idx, point in enumerate(self.path[1:]):
            robot.go_to_pos(point)
            rospy.loginfo("got to point: {} out of {}".format(idx, len(self.path)))
        robot.stop_robot()

    def get_laser_scan(self):
        return self.robot.scan_image

    def get_observation(self):
        orientation_person = np.asarray(self.person.orientation)
        orientation_robot = np.asarray(self.robot.orientation)

        orientation_position = np.append(orientation_person, orientation_robot, self.get_person_position_relative_robot())

        return self.get_laser_scan(), orientation_position

    def __init__(self):
        self.node = rospy.init_node("webots_env", anonymous=True)

        self.supervisor = Supervisor()
        # read the path for robot
        self.path = []
        try:
            with open('data/first', 'rb') as f:
                self.path = pickle.load(f)
        except Exception as e:
            print(e)

        idx_start = 0
        init_pos_person = self.path[idx_start]
        angle_person = self.calculate_angle_using_path(idx_start)
        angle_robot = self.calculate_angle_using_path(idx_start+5)
        self.person = Robot('person', init_pos=self.path[idx_start],  angle=angle_person, supervisor=self.supervisor)
        self.robot = Robot('my_robot', init_pos=self.path[idx_start+5],  angle=angle_robot, supervisor=self.supervisor)
        self.path = self.path[idx_start+5:]
        _thread.start_new_thread(self.path_follower, (self.person,))

        rospy.spin()
        self.observation_space = gym.spaces.Tuple(
                (
                    gym.spaces.Box(low=0, high=1, shape=(50, 100)),
                    gym.spaces.Box(low=0, high=1, shape=(8,))
                )
            )
        # Action space omits the Tackle/Catch actions, which are useful
        # on defense
        self.action_space = gym.spaces.Discrete(8)
        self.min_distance = 1
        self.max_distance = 3.5
        self.number_of_steps = 0
        self.max_numb_steps = 40
        self.distance = 0
        self.reward_range = [-1, 1]

    def __del__(self):
        # todo
        return

    def take_action(self, action):
        self.robot.take_action(action)
        return

    def step(self, action):
        self.number_of_steps += 1
        self.take_action(action)
        reward = self.get_reward()
        ob = self.get_observation()
        episode_over = False
        rel_person = self.get_person_position_relative_robot()
        self.distance = math.hypot(rel_person[0], rel_person[1])
        if self.distance < 0.5:
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
        pos_rel = self.get_person_position_relative_robot()
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
