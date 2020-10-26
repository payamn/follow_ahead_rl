#!/usr/bin/env python

from datetime import datetime

import copy
import traceback

import os, subprocess, time, signal

#from cv_bridge import CvBridge


import gym
import math
import random
# u
import numpy as np
import cv2 as cv

import rospy
# Brings in the SimpleActionClient
import actionlib
# Brings in the .action file and messages used by the move base action
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


from squaternion import quat2euler
from squaternion import euler2quat

from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Point
from geometry_msgs.msg import Point32
from geometry_msgs.msg import TransformStamped
from rosgraph_msgs.msg import Clock

from costmap_converter.msg import ObstacleArrayMsg
from costmap_converter.msg import ObstacleMsg
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist

from gazebo_msgs.srv import SetModelState

import threading

from gym.utils import seeding

import _thread

from squaternion import quat2euler
from squaternion import euler2quat

from simple_pid import PID

import pickle

import logging

logger = logging.getLogger(__name__)


class History():
    def __init__(self, window_size, update_rate, save_rate=10):
        self.idx = 0
        self.update_rate = update_rate
        self.save_rate = save_rate
        self.lock = threading.Lock()
        self.memory_size = int(math.ceil(save_rate/update_rate*window_size)+1)
        self.data = [None for x in range(self.memory_size)]
        self.prev_add_time = rospy.Time.now().to_sec() - 1
        self.window_size = window_size
        self.avg_frame_rate = None
        self.time_data_= []

    def add_element(self, element):
        """
        element: the data that we put inside the history data array
        """

        if abs(rospy.Time.now().to_sec() - self.prev_add_time) < 1./self.save_rate:
            return
        with self.lock:
            self.idx = (self.idx + 1) % self.window_size
            self.prev_add_time = rospy.Time.now().to_sec()
            if self.data[self.idx] is None:
                for idx in range(self.memory_size):
                    self.data[idx] = element
            self.data[self.idx] = element
            if not len(self.time_data_) > 50:
                self.time_data_.append(self.prev_add_time)
                if len(self.time_data_) > 3:
                    prev_t = self.time_data_[0]
                    time_intervals = []
                    for t in self.time_data_[1:]:
                        time_intervals.append(t - prev_t)
                        prev_t = t
                    self.avg_frame_rate = 1.0 / np.average(time_intervals)

    def get_elemets(self):
        return_data = []
        while self.avg_frame_rate is None:
            time.sleep(0.1)
        skip_frames = -int(math.ceil(self.avg_frame_rate / self.update_rate))
        with self.lock:
            index = self.idx #(self.idx - 1)% self.window_size
            if self.window_size * abs(skip_frames) >= self.memory_size:
                rospy.logerr("error in get element memory not enough update rate{} avg_frame_rate{} mem_size {} skipf: {}".format(self.update_rate, self.avg_frame_rate, self.memory_size, skip_frames))
            for i in range (self.window_size):
                return_data.append(self.data[index])
                index = (index + skip_frames) % self.window_size

        return return_data

    def get_latest(self):
        with self.lock:
            return self.data[self.idx]


class Robot():
    def __init__(self, name, max_angular_speed=1, max_linear_speed=1, relative=None, agent_num=None, use_goal=False, use_movebase=False, use_jackal=False, window_size=10, is_testing=False):
        self.name = name
        self.use_jackal = use_jackal
        self.init_node = False
        self.alive = True
        self.prev_call_gazeboros_ = None
        if relative is None:
            relative = self
        self.relative = relative
        self.is_testing = is_testing
        if self.is_testing:
            self.all_pose_ = []
            self.last_time_added = rospy.Time.now().to_sec()
        self.log_history = []
        self.agent_num = agent_num
        self.init_node = True
        self.deleted = False
        self.update_rate_states = 2.0
        self.window_size_history = window_size
        self.current_vel_ = Twist()
        self.goal = {"pos": None, "orientation": None}
        self.use_goal = use_goal
        self.use_movebase = use_movebase
        self.max_angular_vel = max_angular_speed
        self.max_linear_vel = max_linear_speed
        self.max_rel_pos_range = 5.0 # meter
        self.width_laserelement_image = 100
        self.height_laser_image = 50
        self.state_ = {'position':      (None, None),
                       'orientation':   None}
        if self.use_jackal:
            self.cmd_vel_pub =  rospy.Publisher('/{}/jackal_velocity_controller/cmd_vel'.format(name), Twist, queue_size=1)
        else:
            self.cmd_vel_pub =  rospy.Publisher('/{}/cmd_vel'.format(name), Twist, queue_size=1)

        if "tb3" in self.name and self.use_movebase:
            # Create an action client called "move_base" with action definition file "MoveBaseAction"
            self.action_client_ = actionlib.SimpleActionClient('/move_base_{}'.format(self.agent_num),MoveBaseAction)
            # Waits until the action server has started up and started listening for goals.
            self.action_client_.wait_for_server(rospy.rostime.Duration(0.4))
        else:
            self.action_client_ = None

        if "person" is self.name:
            self.angular_pid = PID(0.5, 0, 0.03, setpoint=0)
            self.linear_pid = PID(1.0, 0, 0.05, setpoint=0)
        else:
            self.angular_pid = PID(2.5, 0, 0.03, setpoint=0)
            self.linear_pid = PID(2.5, 0, 0.05, setpoint=0)
        self.pos_history = History(self.window_size_history, self.update_rate_states)
        self.orientation_history = History(self.window_size_history, self.update_rate_states)
        self.velocity_history = History(self.window_size_history, self.update_rate_states)
        self.is_collided = False
        self.is_pause = False
        self.reset = False
        self.scan_image = None

    def calculate_ahead(self, distance):
        x = self.state_['position'][0] + math.cos(self.state_["orientation"]) * distance
        y = self.state_['position'][1] + math.sin(self.state_["orientation"]) * distance
        return (x,y)


    def movebase_cancel_goals(self):
        self.action_client_.cancel_all_goals()
        self.stop_robot()

    def movebase_client_goal(self, goal_pos, goal_orientation):
       # Creates a new goal with the MoveBaseGoal constructor
        move_base_goal = MoveBaseGoal()
        move_base_goal.target_pose.header.frame_id = "tb3_{}/odom".format(self.agent_num)
        move_base_goal.target_pose.header.stamp = rospy.Time.now()
        move_base_goal.target_pose.pose.position.x = goal_pos[0]
        move_base_goal.target_pose.pose.position.y = goal_pos[1]
        quaternion_rotation = euler2quat(0, goal_orientation, 0)

        move_base_goal.target_pose.pose.orientation.x = quaternion_rotation[3]
        move_base_goal.target_pose.pose.orientation.y = quaternion_rotation[1]
        move_base_goal.target_pose.pose.orientation.z = quaternion_rotation[2]
        move_base_goal.target_pose.pose.orientation.w = quaternion_rotation[0]

       # Sends the move_base_goal to the action server.
        self.action_client_.send_goal(move_base_goal)
       # Waits for the server to finish performing the action.
       #wait = self.action_client_.wait_for_result(rospy.rostime.Duration(0.4))
       # If the result doesn't arrive, assume the Server is not available
        # if not wait:
        #     rospy.logerr("Action server not available!")
        # else:
        # # Result of executing the action
        #     return self.action_client_.get_result()

    def get_pos(self):
        counter_problem = 0
        while self.state_['position'] is None:
            if self.reset:
                return (None, None)
            if counter_problem > 20:
                rospy.logdebug("waiting for pos to be available {}/{}".format(counter_problem/10, 20))
            time.sleep(0.001)
            counter_problem += 1
            if counter_problem > 200:
                raise Exception('Probable shared memory issue happend')

        return self.state_['position']



    def get_orientation(self):
        counter_problem = 0
        while self.state_['orientation'] is None:
            if self.reset:
                return None
            if counter_problem > 20:
                rospy.logdebug("waiting for pos to be available {}/{}".format(counter_problem/10, 20))
            time.sleep(0.001)
            counter_problem += 1
            if counter_problem > 200:
                raise Exception('Probable shared memory issue happend')
        return self.state_['orientation']

    def is_current_state_ready(self):
        return (self.state_['position'][0] is not None)

    def is_observation_ready(self):
        return (self.pos_history.avg_frame_rate is not None and\
                self.orientation_history.avg_frame_rate is not None and\
                self.velocity_history.avg_frame_rate is not None)

    def update(self, init_pose):
        self.alive = True
        self.goal = {"pos": None, "orientation": None}
        if "person" is self.name:
            self.angular_pid = PID(0.5, 0, 0.03, setpoint=0)
            self.linear_pid = PID(1.0, 0, 0.05, setpoint=0)
        else:
            self.angular_pid = PID(2.5, 0, 0.03, setpoint=0)
            self.linear_pid = PID(2.5, 0, 0.05, setpoint=0)
        self.pos_history = History(self.window_size_history, self.update_rate_states)
        self.orientation_history = History(self.window_size_history, self.update_rate_states)
        self.velocity_history = History(self.window_size_history, self.update_rate_states)
        self.velocity_history.add_element((0,0))
        self.pos_history.add_element((init_pose["pos"][0],init_pose["pos"][1]))
        self.orientation_history.add_element(init_pose["orientation"])
        self.log_history = []
        if self.is_testing:
            self.all_pose_ = []

        #self.prev_call_gazeboros_ = None
        #self.is_collided = False
        self.is_pause = False
        self.reset = False

    def add_log(self, log):
        self.log_history.append(log)

    def remove(self):
        self.reset = True

    def set_state(self, state):
        self.state_["position"] = state["position"]
        self.state_["orientation"] = state["orientation"]
        self.state_["velocity"] = state["velocity"]

        self.orientation_history.add_element(state["orientation"])
        self.pos_history.add_element(state["position"])
        self.velocity_history.add_element(state["velocity"])
        if self.is_testing and abs (rospy.Time.now().to_sec()- self.last_time_added) > 0.01:
            self.all_pose_.append(self.state_.copy())
            self.last_time_added = rospy.Time.now().to_sec()


    def get_velocity(self):
        return self.velocity_history.get_latest()

    def pause(self):
        self.is_pause = True
        self.stop_robot()

    def resume(self):
        self.is_pause = False

    def take_action(self, action):
        if self.is_pause:
            return

        if self.use_goal:
            pos = GazeborosEnv.denormalize(action[0:2], self.max_rel_pos_range)
            pos_global = GazeborosEnv.get_global_position(pos, self.relative)
            self.goal["orientation"] = self.get_orientation()
            self.goal["pos"] = pos_global
            if self.use_movebase:
                #orientation = GazeborosEnv.denormalize(action[2], math.pi)
                self.movebase_client_goal(pos_global, self.goal["orientation"])
        else:
            linear_vel = max(min(action[0]*self.max_linear_vel, self.max_linear_vel), -self.max_linear_vel)
            angular_vel = max(min(action[1]*self.max_angular_vel, self.max_angular_vel), -self.max_angular_vel)

            cmd_vel = Twist()
            cmd_vel.linear.x = linear_vel #float(self.current_vel_.linear.x -(self.current_vel_.linear.x - linear_vel)*0.9)
            cmd_vel.angular.z = angular_vel #-float(self.current_vel_.angular.z - (self.current_vel_.angular.z - angular_vel)*0.9)
            self.current_vel_ = cmd_vel
            self.cmd_vel_pub.publish(cmd_vel)


    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def angle_distance_to_point(self, pos):
        current_pos = self.get_pos()
        if current_pos[0] is None:
            return None, None
        angle = math.atan2(pos[1] - current_pos[1], pos[0] - current_pos[0])
        distance = math.hypot(pos[0] - current_pos[0], pos[1] - current_pos[1])
        angle = (angle - self.state_["orientation"] + math.pi) % (math.pi * 2) - math.pi
        return angle, distance

    def publish_cmd_vel(self, linear, angular):
        cmd_vel = Twist()
        angular_vel = min(max(angular, -self.max_angular_vel),self.max_angular_vel)
        linear_vel = min(max(linear, 0), self.max_linear_vel)
        cmd_vel.linear.x = float(linear_vel)
        cmd_vel.angular.z = float(angular_vel)
        self.cmd_vel_pub.publish(cmd_vel)

    def use_selected_person_mod(self, person_mode):
        while person_mode<=6:
            if self.is_pause:
                self.stop_robot()
                return
            if self.reset:
                self.stop_robot()
                return
            angular_vel = 0
            linear_vel = 0
            if person_mode == 0:
                linear_vel = self.max_linear_vel
            if person_mode == 1:
                #linear_vel = self.max_linear_vel * random.random()
                linear_vel = self.max_linear_vel * 0.35
            elif person_mode == 2:
                linear_vel = self.max_linear_vel/2
                angular_vel = self.max_angular_vel/6
            elif person_mode == 3:
                linear_vel = self.max_linear_vel/2
                angular_vel = -self.max_angular_vel/6
            elif person_mode == 4:
                linear_vel, angular_vel = self.get_velocity()
                linear_vel = linear_vel - (linear_vel - (random.random()/2 + 0.5))/2.
                angular_vel = -self.max_angular_vel/6
            elif person_mode == 5:
                linear_vel, angular_vel = self.get_velocity()
                linear_vel = linear_vel - (linear_vel - (random.random()/2 + 0.5))/2.
                angular_vel = self.max_angular_vel/6
            elif person_mode == 6:
                linear_vel, angular_vel = self.get_velocity()
                linear_vel = linear_vel - (linear_vel - (random.random()/2 + 0.5))/2.
                angular_vel = angular_vel - (angular_vel - (random.random()-0.5)*2)/2.
            self.publish_cmd_vel(linear_vel, angular_vel)
            time.sleep(0.002)

    def go_to_goal(self):
        while True:
            if self.reset:
                return
            while self.goal["pos"] is None:
                time.sleep(0.1)
                continue
            diff_angle, distance = self.angle_distance_to_point(self.goal["pos"])
            time_prev = rospy.Time.now().to_sec()
            while not distance < 0.1 and abs(rospy.Time.now().to_sec() - time_prev) < 5:
                if self.is_pause:
                    self.stop_robot()
                    return
                if self.reset:
                    self.stop_robot()
                    return
                diff_angle, distance = self.angle_distance_to_point(self.goal["pos"])
                if distance is None:
                    return

                if self.reset:
                    return

                angular_vel = -min(max(self.angular_pid(diff_angle), -self.max_angular_vel),self.max_angular_vel)
                linear_vel = min(max(self.linear_pid(-distance), 0), self.max_linear_vel)
                linear_vel = linear_vel * math.pow((abs(math.pi - abs(diff_angle))/math.pi), 1.5)

                self.publish_cmd_vel(linear_vel, angular_vel)
                time.sleep(0.01)
            self.stop_robot()

    def go_to_pos(self, pos, stop_after_getting=False):
        if self.is_pause:
            self.stop_robot()
            return
        if self.reset:
            return

        diff_angle, distance = self.angle_distance_to_point(pos)
        if distance is None:
            print (self.get_pos())
            return
        time_prev = rospy.Time.now().to_sec()
        while not distance < 0.2 and abs(rospy.Time.now().to_sec() - time_prev) < 5:
            if self.is_pause:
                self.stop_robot()
                return
            if self.reset:
                return
            diff_angle, distance = self.angle_distance_to_point(pos)
            if distance is None:
                return

            if self.reset:
                return

            angular_vel = -min(max(self.angular_pid(diff_angle), -self.max_angular_vel),self.max_angular_vel)
            linear_vel = min(max(self.linear_pid(-distance), 0), self.max_linear_vel)
            linear_vel = linear_vel * math.pow((abs(math.pi - abs(diff_angle))/math.pi), 2)

            self.publish_cmd_vel(linear_vel, angular_vel)
            time.sleep(0.01)

        if stop_after_getting:
            self.stop_robot()

    def get_goal(self):
        counter_problem = 0
        while self.goal["pos"] is None:
            if self.reset:
                return (None, None)
            if counter_problem > 20:
                rospy.logwarn("waiting for goal to be available {}/{}".format(counter_problem/10, 20))
            time.sleep(0.01)
            counter_problem += 1
            if counter_problem > 200:
                raise Exception('Probable shared memory issue happend')
        # if not self.use_movebase:
        #     pos = GazeborosEnv.get_global_position(self.goal["pos"], self)
        #     goal = {"pos":pos, "orientation":None}
        # else:
        #     goal = self.goal

        return self.goal


    def get_pos(self):
        counter_problem = 0
        while self.state_['position'] is None:
            if self.reset:
                return (None, None)
            if counter_problem > 20:
                rospy.logwarn("waiting for pos to be available {}/{}".format(counter_problem/10, 20))
            time.sleep(0.01)
            counter_problem += 1
            if counter_problem > 200:
                raise Exception('Probable shared memory issue happend')

        return self.state_['position']

    def get_laser_image(self):
        return np.expand_dims(self.scan_image, axis=2)

class GazeborosEnv(gym.Env):

    def __init__(self, is_evaluation=False):

        self.is_evaluation_ = is_evaluation

        # self.bridge = CvBridge()
        # self.image_pub = rospy.Publisher("image_observation", Image)
        # self.image_pub_gt = rospy.Publisher("image_observation_gt", Image)

        self.is_reseting = True
        self.use_path = True
        self.use_jackal = True
        self.lock = _thread.allocate_lock()
        self.path_follower_test_settings = {0:(0,0, "straight",False), 1:(2,0, "right", False), 2:(3,0, "left", False),\
                3:(1,4, "straight_Behind", False), 4:(2,3, "right_behind", False), 5:(3,3, "left_behind", False), 6:(7,2, "traj_1", True, True),\
                7:(7, 12, "traj_2", True, True), 8:(7, 43, "traj_3", True),\
                9:(2,1, "right_left", False), 10:(2,2, "right_right", False),\
                11:(3,1, "left_left", False), 12:(3,2, "left_right", False)\
                }
        #self.path_follower_test_settings = {0:(7, 43, "traj_3", True)#(7,2, "traj_1", True, True), 1:(7, 12, "traj_2", True, True)}

        self.is_testing = False
        self.small_window_size = False
        self.use_predifined_mode_person = True
        self.use_goal = True
        self.use_orientation_in_observation = True


        self.collision_distance = 0.3
        self.best_distance = 1.5
        self.robot_mode = 0
        self.window_size = 10
        self.use_movebase = True
        self.use_reachability = False

        self.path_follower_current_setting_idx = 0
        self.use_supervise_action = False
        self.mode_person = 0
        self.use_noise = True
        self.is_use_test_setting = False
        self.use_reverse = True
        if self.small_window_size:
            self.window_size = 5
        if self.is_testing:
            self.use_noise = False
            self.use_reverse = False
            self.is_use_test_setting = True


        self.fallen = False
        self.is_max_distance = False
        self.use_random_around_person_ = False
        self.max_mod_person_ = 7
        self.wait_observation_ = 0

        # being use for observation visualization
        self.center_pos_ = (0, 0)
        self.colors_visualization = cv.cvtColor(cv.applyColorMap(np.arange(0, 255, dtype=np.uint8), cv.COLORMAP_WINTER), cv.COLOR_BGR2RGB).reshape(255,3).tolist()
        self.color_index = 0
        self.first_call_observation = True

        self.test_simulation_ = False

        observation_dimentation = 46
        if self.use_orientation_in_observation:
            observation_dimentation += 1

        if self.small_window_size:
            observation_dimentation -= 20

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(observation_dimentation,))
        self.current_obsevation_image_ = np.zeros([2000,2000,3])
        self.current_obsevation_image_.fill(255)

        self.prev_action = (0, 0)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.min_distance = 1
        self.max_distance = 2.5
        if self.test_simulation_ or self.is_evaluation_:
           self.max_numb_steps = 80
        elif self.is_use_test_setting:
           self.max_numb_steps = 100
        else:
            self.max_numb_steps = 80
        self.reward_range = [-1, 1]
        self.reachabilit_value = None
        if self.use_reachability:
            with open('data/reachability.pkl', 'rb') as f:
                self.reachabilit_value = pickle.load(f)

    def get_test_path_number(self):
        rospy.loginfo("current path idx: {}".format(self.path_follower_current_setting_idx))
        return self.path_follower_test_settings[self.path_follower_current_setting_idx][2]

    def use_test_setting(self):
        self.is_use_test_setting = True


    def set_agent(self, agent_num):
        try:
            self.node = rospy.init_node('gym_gazeboros_{}'.format(agent_num))
        except Exception as e:
            rospy.logerr("probably already init in another node {}".format(e))
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state_sp = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.agent_num = agent_num
        self.obstacle_pub_ =  rospy.Publisher('/move_base_node_{}/TebLocalPlannerROS/obstacles'.format(self.agent_num), ObstacleArrayMsg, queue_size=1)
        self.create_robots()

        self.path = {}
        self.paths = []
        self.log_file = None
        try:
            with open('data/person_trajectories_rl.pkl', 'rb') as f:
                paths = pickle.load(f)
                for path in paths:
                    angle_person = path['start_person']['orientation']
                    for angle in [x for x in range(0, 360, 10)]:
                        for angle_robot_person in [x for x in range(0, 360, 90)]:
                            path_angle = path.copy()
                            angle_from_person = np.deg2rad(angle) + angle_person
                            angle_person_robot = np.deg2rad(angle_robot_person) + angle_person
                            path_angle['start_robot']['pos'] = (path_angle['start_person']['pos'][0] + math.cos(angle_from_person)*2, path_angle['start_person']['pos'][1] + math.sin(angle_from_person)*2)

                            path_angle['start_robot']['orientation'] = angle_person_robot
                            path_angle['name'] = path['name'] + " " + str(angle) +" " + str(angle_robot_person)
                            self.paths.append(path_angle)

                self.path_idx = -1
                self.path = self.paths[self.path_idx]
        except Exception as e:
            print("error happend in writing {}".format(e))

        self.agent_num = agent_num

        self.state_cb_prev_time = None
        self.model_states_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_cb)

        with self.lock:
            self.init_simulator()

    def model_states_cb(self,  states_msg):
        for model_idx in range(len(states_msg.name)):
            found = False
            for robot in [self.robot, self.person]:
                if states_msg.name[model_idx] == robot.name:
                    found = True
                    break
            if not found:
                continue
            pos = states_msg.pose[model_idx]
            euler = quat2euler(pos.orientation.x, pos.orientation.y, pos.orientation.z, pos.orientation.w)
            orientation = euler[0]
            fall_angle = np.deg2rad(90)
            if abs(abs(euler[1]) - fall_angle)< 0.1 or abs(abs(euler[2]) - fall_angle)<0.1:
                self.fallen = True
            # get velocity
            twist = states_msg.twist[model_idx]
            linear_vel = twist.linear.x
            angular_vel = twist.angular.z
            pos_x = pos.position.x
            pos_y = pos.position.y
            state = {}
            state["velocity"] = (linear_vel, angular_vel)
            state["position"] = (pos_x, pos_y)
            state["orientation"] = orientation
            robot.set_state(state)
            if self.use_movebase and robot.name == self.person.name:
                obstacle_msg_array = ObstacleArrayMsg()
                obstacle_msg_array.header.stamp = rospy.Time.now()
                obstacle_msg_array.header.frame_id = "tb3_{}/odom".format(self.agent_num)
                obstacle_msg = ObstacleMsg()
                obstacle_msg.header = obstacle_msg_array.header
                obstacle_msg.id = 0
                for x in range (5):
                    for y in range (5):
                        point = Point32()
                        point.x = pos.position.x + (x-2)*0.1
                        point.y = pos.position.y + (y-2)*0.1
                        point.z = pos.position.z
                        obstacle_msg.polygon.points.append(point)
                obstacle_msg.orientation.x = pos.orientation.x
                obstacle_msg.orientation.y = pos.orientation.y
                obstacle_msg.orientation.z = pos.orientation.z
                obstacle_msg.orientation.w = pos.orientation.w
                obstacle_msg.velocities.twist.linear.x = twist.linear.x
                obstacle_msg.velocities.twist.angular.z = twist.linear.z
                obstacle_msg_array.obstacles.append(obstacle_msg)
                self.obstacle_pub_.publish(obstacle_msg_array)

    def create_robots(self):

        self.person = Robot('person_{}'.format(self.agent_num),
                            max_angular_speed=1, max_linear_speed=.6, agent_num=self.agent_num, window_size=self.window_size, is_testing=self.is_testing)

        relative = self.person

        if self.use_goal:
            relative = self.person
        self.robot = Robot('tb3_{}'.format(self.agent_num),
                            max_angular_speed=1.8, max_linear_speed=0.8, relative=relative, agent_num=self.agent_num, use_goal=self.use_goal, use_movebase=self.use_movebase ,use_jackal=self.use_jackal, window_size=self.window_size, is_testing=self.is_testing)

    def find_random_point_in_circle(self, radious, min_distance, around_point):
        max_r = 2
        r = (radious - min_distance) * math.sqrt(random.random()) + min_distance
        theta = random.random() * 2 * math.pi
        x = around_point[0] + r * math.cos(theta)
        y = around_point[1] + r * math.sin(theta)
        return (x, y)

    def set_mode_person_based_on_episode_number(self, episode_number):
        if episode_number < 500:
            self.mode_person = 0
        elif episode_number < 510:
            self.mode_person = 1
        elif episode_number < 700:
            self.mode_person = 3
        elif episode_number < 900:
            self.mode_person = 5
        elif episode_number < 1000:
            self.mode_person = 6
        else:
            #self.mode_person = 7
            if random.random()>0.5:
                self.mode_person = 7
            else:
                self.mode_person = random.randint(0, 6)

    def get_init_pos_robot_person(self):
        if self.is_evaluation_:
            idx_start = 0
        elif self.is_use_test_setting:
            idx_start = self.path_follower_test_settings[self.path_follower_current_setting_idx][1]
        else:
            idx_start = random.randint(0, len(self.path["points"]) - 20)
        self.current_path_idx = idx_start
        if not self.is_use_test_setting and self.use_reverse and random.random() > 0.5:
            self.path["points"].reverse()

        if self.is_evaluation_:
            init_pos_person = self.path["start_person"]
            init_pos_robot = self.path["start_robot"]
        elif self.is_use_test_setting and not self.path_follower_test_settings[self.path_follower_current_setting_idx][3]:
            init_pos_person = {"pos": (0, 0), "orientation":0}
            mode = self.path_follower_test_settings[self.path_follower_current_setting_idx][1]
            if mode == 0:
                orinetation_person_rob = 0
            elif mode == 1:
                orinetation_person_rob = -math.pi /4.
            elif mode == 2:
                orinetation_person_rob = math.pi /4.
            elif mode == 3:
                orinetation_person_rob = -math.pi
            else:
                orinetation_person_rob = math.pi/8*7
            pos_robot = (1.5*math.cos(orinetation_person_rob), 1.5*math.sin(orinetation_person_rob))
            init_pos_robot = {"pos": pos_robot, "orientation":0}

        elif not self.use_path:
            init_pos_person = {"pos": (0, 0), "orientation": random.random()*2*math.pi - math.pi}
            ahead_person = (init_pos_person['pos'][0] + math.cos(init_pos_person["orientation"]) * 2, init_pos_person['pos'][1] + math.sin(init_pos_person["orientation"]) * 2)
            random_pos_robot = self.find_random_point_in_circle(1.5, 2.5, init_pos_person["pos"])
            init_pos_robot = {"pos": random_pos_robot,\
                              "orientation": init_pos_person["orientation"]}#random.random()*2*math.pi - math.pi}#self.calculate_angle_using_path(idx_start)}
        elif self.use_random_around_person_:
            init_pos_person = {"pos": self.path["points"][idx_start], "orientation": self.calculate_angle_using_path(idx_start)}
            init_pos_robot = {"pos": self.find_random_point_in_circle(1.5, 1, self.path["points"][idx_start]),\
                              "orientation": random.random()*2*math.pi - math.pi}#self.calculate_angle_using_path(idx_start)}
        else:
            init_pos_person = {"pos": self.path["points"][idx_start], "orientation": self.calculate_angle_using_path(idx_start)}
            if self.is_use_test_setting and len(self.path_follower_test_settings[self.path_follower_current_setting_idx])>4 and self.path_follower_test_settings[self.path_follower_current_setting_idx][4] :
                orinetation_person_rob = math.pi/2.2
                pos_robot = (self.path["points"][idx_start][0] + 2*math.cos(orinetation_person_rob+init_pos_person["orientation"]), self.path["points"][idx_start][1] + 2*math.sin(orinetation_person_rob+init_pos_person["orientation"]))
                init_pos_robot = {"pos": pos_robot, "orientation":self.calculate_angle_using_path(idx_start+5)}
            else:

                idx_robot = idx_start + 1
                while (math.hypot(self.path["points"][idx_robot][1] - self.path["points"][idx_start][1],
                                  self.path["points"][idx_robot][0] - self.path["points"][idx_start][0]) < 1.6):
                    idx_robot += 1

                init_pos_robot = {"pos": self.path["points"][idx_robot],\
                                  "orientation": self.calculate_angle_using_path(idx_robot)}
                if not self.is_testing:
                    init_pos_robot["pos"] = (init_pos_robot["pos"][0]+ random.random()-0.5, \
                            init_pos_robot["pos"][1]+ random.random()-0.5)
                    init_pos_robot["orientation"] = GazeborosEnv.wrap_pi_to_pi(init_pos_robot["orientation"] + random.random()-0.5)

        return init_pos_robot, init_pos_person

    def set_pos(self, name, pose):
        set_model_msg = ModelState()
        set_model_msg.model_name = name
        self.prev_action = (0,0)
        quaternion_rotation = euler2quat(0, pose["orientation"], 0)

        set_model_msg.pose.orientation.x = quaternion_rotation[3]
        set_model_msg.pose.orientation.y = quaternion_rotation[1]
        set_model_msg.pose.orientation.z = quaternion_rotation[2]
        set_model_msg.pose.orientation.w = quaternion_rotation[0]

        if self.use_jackal and "tb3" in name:
            set_model_msg.pose.position.z = 2.6 * self.agent_num + 0.1635
        else:
            set_model_msg.pose.position.z = 2.6 * self.agent_num + 0.099
        set_model_msg.pose.position.x = pose["pos"][0]
        set_model_msg.pose.position.y = pose["pos"][1]
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state_sp(set_model_msg)

    def init_simulator(self):

        self.number_of_steps = 0
        rospy.loginfo("init simulation called")
        self.is_pause = True

        init_pos_robot, init_pos_person = self.get_init_pos_robot_person()
        self.center_pos_ = init_pos_person["pos"]
        self.color_index = 0
        self.fallen = False
        self.is_max_distance = False
        self.first_call_observation = True

        self.current_obsevation_image_.fill(255)
        if self.use_movebase:
            self.robot.movebase_cancel_goals()
        rospy.sleep(0.5)
        self.person.stop_robot()
        self.robot.stop_robot()
        # if self.use_movebase:
        #     self.prev_action = (0,0, 0)
        # else:
        self.prev_action = (0,0)
        self.set_pos(self.robot.name, init_pos_robot)
        self.set_pos(self.person.name, init_pos_person)

        self.robot.update(init_pos_robot)
        self.person.update(init_pos_person)

        self.path_finished = False
        self.position_thread = threading.Thread(target=self.path_follower, args=(self.current_path_idx, self.robot,))
        self.position_thread.daemon = True

        self.is_reseting = False
        self.position_thread.start()
        self.wait_observation_ = 0

        self.is_reseting = False
        self.robot.reset = False
        self.person.reset = False

        # self.resume_simulator()
        rospy.loginfo("init simulation finished")
        self.is_pause = False

    def pause(self):
        self.is_pause = True
        self.person.pause()
        self.robot.pause()

    def resume_simulator(self):
        rospy.loginfo("resume simulator")
        self.is_pause = False
        self.person.resume()
        self.robot.resume()
        rospy.loginfo("resumed simulator")

    def calculate_angle_using_path(self, idx):
        return math.atan2(self.path["points"][idx+1][1] - self.path["points"][idx][1], self.path["points"][idx+1][0] - self.path["points"][idx][0])

    @staticmethod
    def denormalize(value, max_val):
        if type(value) == tuple or type(value) == list:
            norm_val = [float(x) * max_val for x in value]
        else:
            norm_val = value * float(max_val)
        return norm_val

    @staticmethod
    def normalize(value, max_val, zero_to_one=None):
        if type(value) == tuple or type(value) == list:
            norm_val = [x/float(max_val) for x in value]
        else:
            norm_val = value/float(max_val)
        if zero_to_one is not None:
            if type(value) == tuple or type(value) == list:
                norm_val = [(x + 1)/2 for x in norm_val]
            else:
                norm_val = (norm_val + 1)/2.

        return norm_val

    @staticmethod
    def get_global_position(pos_goal, center):
        while not center.is_current_state_ready():
            if center.reset:
                rospy.logwarn("reseting so return none in rel pos rel: {} center".format(relative.is_current_state_ready(), center.is_current_state_ready()))
                return (None, None)
            time.sleep(0.01)
            rospy.logwarn ("waiting for observation to be ready")
        #relative_orientation = relative.state_['orientation']
        center_pos = np.asarray(center.state_['position'])
        center_orientation = center.state_['orientation']

        #pos = [x * 5 for x in pos_goal]

        relative_pos = np.asarray(pos_goal)

        # transform the relative to center coordinat
        rotation_matrix = np.asarray([[np.cos(center_orientation), np.sin(center_orientation)], [-np.sin(center_orientation), np.cos(center_orientation)]])
        relative_pos = np.matmul(relative_pos, rotation_matrix)
        global_pos = np.asarray(relative_pos + center_pos)
        return global_pos

    @staticmethod
    def get_global_position_orientation(pos_goal, orientation_goal, center):
        while not center.is_current_state_ready():
            if center.reset:
                rospy.logwarn("reseting so return none in rel pos rel: {} center".format(relative.is_current_state_ready(), center.is_current_state_ready()))
                return (None, None)
            time.sleep(0.01)
            rospy.logwarn ("waiting for observation to be ready")
        #relative_orientation = relative.state_['orientation']
        center_pos = np.asarray(center.state_['position'])
        center_orientation = center.state_['orientation']

        #pos = [x * 5 for x in pos_goal]

        relative_pos = np.asarray(pos_goal)
        relative_pos2 = np.asarray((relative_pos[0] +math.cos(orientation_goal) , relative_pos[1] + math.sin(orientation_goal)))

        # transform the relative to center coordinat
        rotation_matrix = np.asarray([[np.cos(center_orientation), np.sin(center_orientation)], [-np.sin(center_orientation), np.cos(center_orientation)]])
        relative_pos = np.matmul(relative_pos, rotation_matrix)
        relative_pos2 = np.matmul(relative_pos2, rotation_matrix)
        global_pos = np.asarray(relative_pos + center_pos)
        global_pos2 = np.asarray(relative_pos2 + center_pos)
        new_orientation = np.arctan2(global_pos2[1]-global_pos[1], global_pos2[0]-global_pos[0])
        return global_pos, new_orientation


    @staticmethod
    def wrap_pi_to_pi(angle):
        while angle > math.pi:
            angle -= 2*math.pi
        while angle < - math.pi:
            angle += 2*math.pi
        return angle

    @staticmethod
    def get_relative_heading_position(relative, center):
        while not relative.is_current_state_ready() or not center.is_current_state_ready():
            if relative.reset:
                rospy.logwarn("reseting so return none in rel pos rel: {} center".format(relative.is_current_state_ready(), center.is_current_state_ready()))
                return (None, None)
            time.sleep(0.1)
            rospy.loginfo ("waiting for observation to be ready heading pos")
        relative_orientation = relative.state_['orientation']
        center_pos = np.asarray(center.state_['position'])
        center_orientation = center.state_['orientation']

        # transform the relative to center coordinat
        relative_pos = np.asarray(relative.state_['position'] - center_pos)
        relative_pos2 = np.asarray((relative_pos[0] +math.cos(relative_orientation) , relative_pos[1] + math.sin(relative_orientation)))
        rotation_matrix = np.asarray([[np.cos(-center_orientation), np.sin(-center_orientation)], [-np.sin(-center_orientation), np.cos(-center_orientation)]])
        relative_pos = np.matmul(relative_pos, rotation_matrix)
        relative_pos2 = np.matmul(relative_pos2, rotation_matrix)
        angle_relative = np.arctan2(relative_pos2[1]-relative_pos[1], relative_pos2[0]-relative_pos[0])
        return -angle_relative, relative_pos

    @staticmethod
    def get_relative_position(pos, center):
        while not center.is_current_state_ready():
            if center.reset:
                rospy.loginfo("reseting so return none in rel pos rel: {} center".format(relative.is_current_state_ready(), center.is_current_state_ready()))
                return (None, None)
            time.sleep(0.01)
            rospy.loginfo("waiting for observation to be ready relative pos")
        #relative_orientation = relative.state_['orientation']
        center_pos = np.asarray(center.state_['position'])
        center_orientation = center.state_['orientation']

        relative_pos = np.asarray(pos)
        # transform the relative to center coordinat
        relative_pos = np.asarray(relative_pos - center_pos)
        rotation_matrix = np.asarray([[np.cos(-center_orientation), np.sin(-center_orientation)], [-np.sin(-center_orientation), np.cos(-center_orientation)]])
        relative_pos = np.matmul(relative_pos, rotation_matrix)
        return relative_pos


    def set_robot_to_auto(self):
        self.robot_mode = 1

    """
    the function will check the self.robot_mode:
        0: will not move robot
        1: robot will try to go to a point after person
    """
    def path_follower(self, idx_start, robot):

        counter = 0
        while self.is_pause:
            if self.is_reseting:
                rospy.loginfo( "path follower return as reseting ")
                return
            time.sleep(0.001)
            if counter > 10000:
                rospy.loginfo( "path follower waiting for pause to be false")
                counter = 0
            counter += 1
        rospy.loginfo( "path follower waiting for lock pause:{} reset:{}".format(self.is_pause, self.is_reseting))
        if self.lock.acquire(timeout=10):
            rospy.sleep(1.5)
            rospy.loginfo("path follower got the lock")
            if self.is_use_test_setting:
                mode_person = self.path_follower_test_settings[self.path_follower_current_setting_idx][0]
            elif self.test_simulation_:
                mode_person = -1
            elif self.is_evaluation_:
                mode_person = 2
            elif self.use_predifined_mode_person:
                mode_person = self.mode_person
            else:
                mode_person = random.randint(0, 7)
                #if self.agent_num == 2:
                #    mode_person = random.randint(1, self.max_mod_person_)
                #else:
                #    mode_person = 0
                # if self.agent_num == 0:
                #     mode_person = 5
                # elif self.agent_num == 1:
                #     mode_person = 2
                # elif self.agent_num == 2:
                #     mode_person = 3
                # elif self.agent_num == 3:
                #     mode_person = 7
                # else:
                #     mode_person = random.randint(1, self.max_mod_person_)
            # if mode_person == 0:
            #     person_thread = threading.Thread(target=self.person.go_to_goal, args=())
            #     person_thread.start()
            if self.use_goal and not self.use_movebase:
                self.robot_thread = threading.Thread(target=self.robot.go_to_goal, args=())
                self.robot_thread.start()

            for idx in range (idx_start, len(self.path["points"])-3):
                point = (self.path["points"][idx][0], self.path["points"][idx][1])
                self.current_path_idx = idx
                counter_pause = 0
                while self.is_pause:
                    counter_pause+=1
                    rospy.loginfo("pause in path follower")
                    if self.is_reseting or counter_pause > 200:
                        # if mode_person == 0:
                        #     person_thread.join()
                        self.lock.release()
                        return
                    time.sleep(0.001)
                try:
                    if mode_person <= 6:
                        self.person.use_selected_person_mod(mode_person)
                    else:
                        self.person.go_to_pos(point, stop_after_getting=True)
                        time.sleep(0.001)
                    # person_thread.start()
                    # if self.robot_mode == 1:
                    #     noisy_point = (self.path["points"][idx+3][0] +min(max(np.random.normal(),-0.5),0.5), self.path["points"][idx+3][1] +min(max(np.random.normal(),-0.5),0.5))

                    #     robot_thread = threading.Thread(target=self.robot.go_to_pos, args=(noisy_point,True,))
                    #     robot_thread.start()
                    #     robot_thread.join()

                    # person_thread.join()

                except Exception as e:
                    rospy.logerr("path follower {}, {}".format(self.is_reseting, e))
                    traceback.print_exc()
                    break
                if self.is_reseting:
                    self.person.stop_robot()
                    break
            self.lock.release()
            rospy.loginfo("path follower release the lock")
            self.path_finished = True
        else:
            rospy.loginfo("problem in getting the log in path follower")
        # robot.stop_robot()


    def get_laser_scan(self):
        return self.robot.get_laser_image()

    def get_laser_scan_all(self):
        images = self.robot.scan_image_history.get_elemets()
        counter = 0
        while len(images)!=self.robot.scan_image_history.window_size and counter<250:
            images = self.robot.scan_image_history.get_elemets()
            time.sleep(0.005)
            counter +=1
            if counter > 100:
                rospy.loginfo("wait for laser scan to get filled sec: {}/25".format(counter / 10))
        if counter>=250:
            raise RuntimeError(
                'exception while calling get_laser_scan:')


        images = np.asarray(images)

        return (images.reshape((images.shape[1], images.shape[2], images.shape[0])))



    def get_observation(self):
        # got_laser = False
        # while not got_laser:
        #     try:
        #         laser_all = self.get_laser_scan_all()
        #         got_laser = True
        #     except Exception as e:
        #         rospy.logerr("laser_error reseting")
        #         # self.reset(reset_gazebo = True)
        while self.robot.pos_history.avg_frame_rate is None or self.person.pos_history.avg_frame_rate is None or self.robot.velocity_history.avg_frame_rate is None or self.person.velocity_history.avg_frame_rate is None:
            if self.is_reseting:
                return None
            time.sleep(0.001)
        pos_his_robot = np.asarray(self.robot.pos_history.get_elemets())
        heading_robot = self.robot.state_["orientation"]

        pos_his_person = np.asarray(self.person.pos_history.get_elemets())
        heading_person = self.person.state_["orientation"]

        robot_vel = np.asarray(self.robot.get_velocity())
        person_vel = np.asarray(self.person.get_velocity())
        poses = np.concatenate((pos_his_robot, pos_his_person))
        if self.use_noise:
            poses += np.random.normal(loc=0, scale=0.1, size=poses.shape)
            heading_robot += np.random.normal(loc=0, scale=0.2)
            heading_person += np.random.normal(loc=0, scale=0.2)
            robot_vel += np.random.normal(loc=0, scale=0.1, size=robot_vel.shape)
            person_vel += np.random.normal(loc=0, scale=0.1, size=person_vel.shape)
        heading_relative = GazeborosEnv.wrap_pi_to_pi(heading_robot-heading_person)/(math.pi)
        pos_rel = []
        for pos in (poses):
            relative = GazeborosEnv.get_relative_position(pos, self.robot.relative)
            pos_rel.append(relative)
        pos_history = np.asarray(np.asarray(pos_rel)).flatten()/6.0
        #TODO: make the velocity normalization better
        velocities = np.concatenate((person_vel, robot_vel))/self.robot.max_angular_vel
        if self.use_orientation_in_observation:
            velocities_heading = np.append(velocities, heading_relative)
        else:
            velocities_heading = velocities
        final_ob =  np.append(np.append(pos_history, velocities_heading), self.prev_action)

        return final_ob

    def __del__(self):
        # todo
        return

    def visualize_observation(self):
        observation_image = np.zeros([2000,2000,3])
        observation_image_gt = np.zeros([2000,2000,3])
        observation_image = observation_image.astype(np.uint8)
        observation_image_gt = observation_image_gt.astype(np.uint8)
        observation_image.fill(255)
        observation_image_gt.fill(255)
        while self.robot.pos_history.avg_frame_rate is None or self.person.pos_history.avg_frame_rate is None or self.robot.velocity_history.avg_frame_rate is None or self.person.velocity_history.avg_frame_rate is None:
            if self.is_reseting:
                return None
            time.sleep(0.001)
        pos_his_robot = self.robot.pos_history.get_elemets()
        heading_robot = self.robot.state_["orientation"]

        pos_his_person = self.person.pos_history.get_elemets()
        heading_person = self.person.state_["orientation"]

        heading_relative = GazeborosEnv.wrap_pi_to_pi(heading_robot-heading_person)/(math.pi)
        center_pos = pos_his_robot[-1]
        for pos in pos_his_robot:
            relative = GazeborosEnv.get_relative_position(pos, self.robot)
            pos_rel = GazeborosEnv.to_image_coordinate(relative, (0, 0))
            pos_gt = GazeborosEnv.to_image_coordinate(pos, center_pos)
            observation_image = self.add_circle_observation_to_image(relative, (255, 0, 0), 10, center_pos=(0,0), image=observation_image)
            observation_image_gt = self.add_circle_observation_to_image(pos, (255, 0, 0), 10, center_pos=center_pos, image=observation_image_gt)

        for pos in pos_his_person:
            relative = GazeborosEnv.get_relative_position(pos, self.robot)
            pos_rel = GazeborosEnv.to_image_coordinate(relative, (0, 0))
            pos_gt = GazeborosEnv.to_image_coordinate(pos, center_pos)
            observation_image = self.add_circle_observation_to_image(relative, (0, 255, 0), 10, image = observation_image, center_pos=(0,0))
            observation_image_gt = self.add_circle_observation_to_image(pos, (0, 255, 0), 10, image=observation_image_gt, center_pos=center_pos)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(observation_image, encoding="bgr8"))
        self.image_pub_gt.publish(self.bridge.cv2_to_imgmsg(observation_image_gt, encoding="bgr8"))


    @staticmethod
    def to_image_coordinate(pos, center_pos):
        return (int((pos[0] - center_pos[0])*50+1000), int((pos[1] - center_pos[1])*50+1000))

    def add_line_observation_to_image(self, pos, pos2):
        color = self.colors_visualization[self.color_index]
        pos_image = GazeborosEnv.to_image_coordinate(pos, self.center_pos_)
        pos_image2 = GazeborosEnv.to_image_coordinate(pos2, self.center_pos_)
        if pos_image[0] >2000 or pos_image[0] < 0 or pos_image[1] >2000 or pos_image[1] < 0:
            rospy.logerr("problem with observation: {}".format(pos_image))
            return
        self.new_obsevation_image_ = cv.line(self.new_obsevation_image_, (pos_image[0], pos_image[1]), (pos_image2[0], pos_image2[1]), color, 1)

    def add_triangle_observation_to_image(self, pos, orientation):
        color = self.colors_visualization[self.color_index]
        pos_image = GazeborosEnv.to_image_coordinate(pos, self.center_pos_)
        pos_triangle1 = GazeborosEnv.to_image_coordinate((pos[0]+math.cos(orientation)*0.3, pos[1]+math.sin(orientation)*0.3), self.center_pos_)
        pos_triangle2 = GazeborosEnv.to_image_coordinate((pos[0]+math.cos(orientation+math.pi/2)*0.1, pos[1]+math.sin(orientation+math.pi/2)*0.1), self.center_pos_)
        pos_triangle3 = GazeborosEnv.to_image_coordinate((pos[0]+math.cos(orientation-math.pi/2)*0.1, pos[1]+math.sin(orientation-math.pi/2)*0.1), self.center_pos_)
        poses = [pos_triangle1, pos_triangle2, pos_triangle3]
        print(poses)

        for pos in poses:
            if pos[0] >2000 or pos[0] < 0 or pos[1] >2000 or pos[1] < 0:
                rospy.logerr("problem with observation: {}".format(pos))
                return
        self.new_obsevation_image_ = cv.drawContours(self.new_obsevation_image_, [np.asarray(poses)], 0, color, -1)


    def add_arrow_observation_to_image(self, pos, orientation):
        color = self.colors_visualization[self.color_index]
        pos_image = GazeborosEnv.to_image_coordinate(pos, self.center_pos_)
        pos_image2 = GazeborosEnv.to_image_coordinate((pos[0]+math.cos(orientation)*0.3, pos[1]+math.sin(orientation)*0.3), self.center_pos_)
        if pos_image[0] >2000 or pos_image[0] < 0 or pos_image[1] >2000 or pos_image[1] < 0:
            rospy.logerr("problem with observation: {}".format(pos_image))
            return
        self.new_obsevation_image_ = cv.arrowedLine(self.new_obsevation_image_, (pos_image[0], pos_image[1]), (pos_image2[0], pos_image2[1]), color, 2, tipLength=0.5)

    def add_circle_observation_to_image(self, pos, color, radious, center_pos=None, image=None):
        if image is None:
            image = self.new_obsevation_image_
        if center_pos is None:
            center_pos = self.center_pos_
        pos_image = GazeborosEnv.to_image_coordinate(pos, center_pos)
        if pos_image[0] >2000 or pos_image[0] < 0 or pos_image[1] >2000 or pos_image[1] < 0:
            rospy.logerr("problem with observation: {}".format(pos_image))
            return
        return (cv.circle(image , (pos_image[0], pos_image[1]), radious, color, 2))

    def get_supervised_action(self):
        while not self.person.is_current_state_ready() and not self.is_reseting:
            time.sleep(0.1)
        if self.is_reseting:
            return np.asarray([0,0])

        self.use_supervise_action = True
        pos = self.person.calculate_ahead(1.5)
        pos_person = self.person.get_pos()
        pos_relative = GazeborosEnv.get_relative_position(pos, self.robot.relative)
        pos_person_relative = GazeborosEnv.get_relative_position(pos_person, self.robot.relative)
        print (f"pos pos_person pos_relative [pos], [pos_person] [pos_relative]")
        pos_norm = GazeborosEnv.normalize(pos_relative, self.robot.max_rel_pos_range)
        orientation = GazeborosEnv.normalize(math.atan2(pos_relative[1] - pos_person_relative[1], pos_relative[0] - pos_person_relative[0]), math.pi)
        return np.asarray((pos_norm[0], pos_norm[1], orientation))


    def update_observation_image(self):
        self.new_obsevation_image_ = np.copy(self.current_obsevation_image_)
        robot_pos = self.robot.get_pos()
        robot_orientation = self.robot.get_orientation()
        person_pos = self.person.get_pos()
        person_orientation = self.person.get_orientation()
        if self.use_goal:
            current_goal = self.robot.get_goal()
        if person_orientation is None or robot_orientation is None:
            rospy.logerr("person or robot orientation is None")
            return
        if self.first_call_observation:
            # self.new_obsevation_image_ = self.add_circle_observation_to_image(robot_pos, [152,100,100], 10)
            # self.new_obsevation_image_ = self.add_circle_observation_to_image(person_pos,[0,100,100], 10)
            self.first_call_observation = False
        if self.is_collided():
            self.new_obsevation_image_ = self.add_circle_observation_to_image(robot_pos, [152,200,200], 10)
            self.new_obsevation_image_ = self.add_circle_observation_to_image(person_pos,[200,100,100], 10)
        self.add_arrow_observation_to_image(robot_pos, robot_orientation)
        self.add_triangle_observation_to_image(person_pos, person_orientation)

        if self.use_goal:
            if self.use_movebase:
                goal_orientation = current_goal["orientation"]
            else:
                goal_orientation = robot_orientation
            self.add_circle_observation_to_image(current_goal["pos"], self.colors_visualization[self.color_index], 5)
            #self.add_line_observation_to_image(robot_pos, current_goal["pos"])
        else:
            self.add_line_observation_to_image(robot_pos, person_pos)
        alpha = 0.50
        self.current_obsevation_image_ = cv.addWeighted(self.new_obsevation_image_, alpha, self.current_obsevation_image_, 1 - alpha, 0)



    def get_current_observation_image(self):

        image = self.current_obsevation_image_
        image = image/255.

        return image


    def take_action(self, action):
        self.prev_action = action[:2]
        self.robot.take_action(action)
        if self.wait_observation_ <= 0:
            self.update_observation_image()
            self.wait_observation_ = 7
        self.color_index += 2
        if self.color_index >= len(self.colors_visualization):
            self.color_index = len(self.colors_visualization) - 1
        self.wait_observation_ -= 1
        return

    def is_skip_run(self):
        if self.fallen:
            return True
        else:
            return False

    def is_successful(self):
        if self.is_collided() or self.is_max_distance or self.fallen:
            return False
        else:
            return True

    def step(self, action):
        self.number_of_steps += 1
        self.take_action(action)
        # instead of one reward get all the reward during wait
        # rospy.sleep(0.4)
        sleep_time = 0.10
        rewards = []
        if sleep_time > 0.1:
            for t in range (10):
                rospy.sleep(sleep_time/10.)
                rewards.append(self.get_reward())
                reward = np.mean(rewards)
        else:
             rospy.sleep(sleep_time)
             reward = self.get_reward()
        ob = self.get_observation()
        episode_over = False
        rel_person = GazeborosEnv.get_relative_heading_position(self.robot, self.person)[1]

        distance = math.hypot(rel_person[0], rel_person[1])
        if self.path_finished:
            rospy.loginfo("path finished")
            episode_over = True
        if self.is_collided():
            self.update_observation_image()
            episode_over = True
            rospy.loginfo('collision happened episode over')
            reward -= 0.5
        elif distance > 5:
            self.update_observation_image()
            self.is_max_distance = True
            episode_over = True
            rospy.loginfo('max distance happened episode over')
        elif self.number_of_steps > self.max_numb_steps:
            self.update_observation_image()
            episode_over = True
        if self.fallen:
            episode_over = True
            rospy.loginfo('fallen')
        reward = min(max(reward, -1), 1)
        if self.agent_num == 0:
            rospy.loginfo("action {} reward {}".format(action, reward))
        if episode_over:
            self.person.reset = True
        #reward += 1
        return ob, reward, episode_over, {}

    def is_collided(self):
        rel_person = GazeborosEnv.get_relative_heading_position(self.robot, self.person)[1]
        distance = math.hypot(rel_person[0], rel_person[1])
        if distance < self.collision_distance or self.robot.is_collided:
            return True
        return False

    def get_distance(self):
        _, pos_rel = GazeborosEnv.get_relative_heading_position(self.robot, self.person)
        return math.hypot(pos_rel[0],pos_rel[1])

    def get_angle_person_robot(self):
        _, pos_rel = GazeborosEnv.get_relative_heading_position(self.robot, self.person)
        angle_robot_person = math.atan2(pos_rel[1], pos_rel[0])
        return (GazeborosEnv.wrap_pi_to_pi(angle_robot_person))

    def get_reward(self):
        reward = 0
        angle_robot_person, pos_rel = GazeborosEnv.get_relative_heading_position(self.robot, self.person)
        angle_robot_person = math.atan2(pos_rel[1], pos_rel[0])
        angle_robot_person = np.rad2deg(GazeborosEnv.wrap_pi_to_pi(angle_robot_person))
        distance = math.hypot(pos_rel[0], pos_rel[1])
        # Negative reward for being behind the person
        if self.is_collided():
            reward -= 1
        if distance < 0.5:
            reward = -1.3
        elif abs(distance - self.best_distance) < 0.5:
            reward += 0.5 * (0.5 - abs(distance - self.best_distance))
        elif distance >= self.best_distance+0.5:
            reward -= 0.25 * (distance - (self.best_distance+0.5))
        elif distance < self.best_distance-0.5:
            reward -= (self.best_distance - 0.5 - distance)/(self.best_distance - 0.5)
        if abs(angle_robot_person) < 25:
            reward += 0.5 * (25 - abs(angle_robot_person)) / 25
        else:
            reward -= 0.25 * abs(angle_robot_person) / 180
        if abs(distance - self.best_distance) < 0.5 and abs(angle_robot_person) < 25:
            reward += 0.25

        # if not 90 > angle_robot_person > 0:
        #     reward -= distance/6.0
        # elif self.min_distance < distance < self.max_distance:
        #     reward += 0.1 + (90 - angle_robot_person) * 0.9 / 90
        # elif distance < self.min_distance:
        #     reward -= 1 - distance / self.min_distance
        # else:
        #     reward -= distance / 7.0
        reward = min(max(reward, -1), 1)
        # ToDO check for obstacle
        return reward

    def save_log(self):
        pickle.dump({"person_history":self.person.log_history, "robot_history":self.robot.log_history}, self.log_file)
        self.log_file.close()




    def reset(self, reset_gazebo=False):

        self.is_pause = True
        self.is_reseting = True
        self.robot.reset = True
        self.person.reset = True
        rospy.loginfo("trying to get the lock for reset")
        # if reset_gazebo:
        #     self.reset_gazebo()
        with self.lock:

            rospy.loginfo("got the lock")
            not_init = True
            try:

                if self.is_evaluation_:
                    if self.log_file is not None:
                        pickle.dump({"person_history":self.person.log_history, "robot_history":self.robot.log_history}, self.log_file)
                        self.log_file.close()

                    self.path_idx += 1
                    print ("start path_id: {}".format(self.path_idx))
                    if self.path_idx < len(self.paths)-1:
                        self.path = self.paths[self.path_idx]
                        self.log_file = open(self.path["name"], "wb")
                    else:
                        print ("all done")
                        self.person.stop_robot()
                        exit(0)
                self.init_simulator()
                not_init = False
            except RuntimeError as e:
                rospy.logerr("error happend reseting: {}".format(e))
        if not_init:
            rospy.loginfo("not init so run reset again")
            return (self.reset())
        else:
            rospy.sleep(2)
            return self.get_observation()

    def save_current_path(self):
        all_pos_robot = self.robot.all_pose_
        all_pos_person = self.person.all_pose_
        directory = "data/traj_simulations"
        name = ""
        if self.use_goal:
            if self.use_supervise_action:
                name += "base_"
            else:
                name += "planner_"
        else:
            name += "cmd_"
        name += self.path_follower_test_settings[self.path_follower_current_setting_idx][2]
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, name + ".pkl") , "wb") as f:
            pickle.dump({"robot":all_pos_robot, "person":all_pos_person, "name":name}, f)
        self.robot.all_pose_ = []
        self.person.all_pose_ = []

    def next_setting(self):
        self.save_current_path()
        self.path_follower_current_setting_idx += 1

    def is_finish(self):
        if self.path_follower_current_setting_idx >= len(self.path_follower_test_settings)-1:
            self.save_current_path()
            return True
        return False

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        return

    def calculate_rechability_derivite(self, x, y, v, theta):

        get_idx = lambda x: int(math.floor(x))
        pos_norm = GazeborosEnv.normalize((x, y), self.robot.max_rel_pos_range, True)
        orientation_norm = GazeborosEnv.normalize(theta, math.pi, True)
        velocity_norm = GazeborosEnv.normalize(v, self.robot.max_linear_vel, True)
        x_idx = get_idx(pos_norm[0]*(self.reachabilit_value.shape[0]-1))
        y_idx = get_idx(pos_norm[1]*(self.reachabilit_value.shape[1]-1))
        orientation_idx = get_idx(orientation_norm * (self.reachabilit_value.shape[3] -1))
        v_idx = get_idx(velocity_norm * (self.reachabilit_value.shape[2]-1))

        rospy.loginfo("x: {} y: {} theta {}".format(x_idx, y_idx, orientation_idx))
        v_idx = max(min(v_idx, self.reachabilit_value.shape[2]-2), 0)
        orientation_idx = max(min(orientation_idx, self.reachabilit_value.shape[3]-2), 0)
        x_idx =  max(min(x_idx, self.reachabilit_value.shape[0]-1), 0)
        y_idx =  max(min(y_idx, self.reachabilit_value.shape[1]-1), 0)

        derivative_v = (self.reachabilit_value[x_idx, y_idx, v_idx+1, orientation_idx] -\
                       self.reachabilit_value[x_idx, y_idx, v_idx, orientation_idx])/2


        derivative_theta = (self.reachabilit_value[x_idx, y_idx, v_idx, orientation_idx+1] -\
                       self.reachabilit_value[x_idx, y_idx, v_idx, orientation_idx])/2


        rospy.loginfo("x: {} y: {} theta {}".format(x_idx, y_idx, orientation_idx))
        return derivative_v, derivative_theta, self.reachabilit_value[x_idx, y_idx, v_idx, orientation_idx]

    def reachability_action(self):
        relative = GazeborosEnv.get_relative_position(self.robot.get_pos(), self.person)
        orientation = GazeborosEnv.wrap_pi_to_pi(self.robot.get_orientation() - self.person.get_orientation())
        print (np.rad2deg(orientation), np.rad2deg(self.person.get_orientation()), np.rad2deg(self.robot.get_orientation()) )
        velocity = self.robot.get_velocity()[0]
        derivative_v, derivative_theta, v = self.calculate_rechability_derivite(relative[0], relative[1], velocity, orientation)
        rospy.loginfo("d_v: {:0.5f} W: {:0.5f} v {:0.1f}".format(derivative_v, derivative_theta, v))
        action = [0,0]
        if v<1:
            if derivative_v > 0:
                action[0] = 1
            else:
                action[0] = -1
            if derivative_theta > 0:
                action[1] = 1
            else:
                action[1] = -1

        return action






#def read_bag():
#    gazeboros_n = GazeborosEnv()
#    gazeboros_n.set_agent(0)
#
#    while gazeboros_n.robot.prev_call_gazeboros_ is None or rospy.Time.now().to_sec() - gazeboros_n.robot.prev_call_gazeboros_ < 5:
#        rospy.sleep(0.1)
#    gazeboros_n.save_log()
#    print("done")

#read_bag()

def test():
    gazeboros_env = GazeborosEnv()
    gazeboros_env.set_agent(0)
    step = 0
    while (True):
        step +=1
        #action = gazeboros_env.get_supervised_action()
        #action = gazeboros_env.reachability_action()
        #gazeboros_env.step(action)
        rel_person = GazeborosEnv.get_relative_heading_position(gazeboros_env.robot, gazeboros_env.person)[1]
        relative_pos2 = GazeborosEnv.get_relative_position(gazeboros_env.robot.get_pos(), gazeboros_env.robot.relative)
        orientation1 = np.rad2deg(np.arctan2(rel_person[1], rel_person[0]))
        distance = math.hypot(relative_pos2[0], relative_pos2[1])

        heading_robot = gazeboros_env.robot.state_["orientation"]
        heading_person = gazeboros_env.person.state_["orientation"]
        heading_relative = GazeborosEnv.wrap_pi_to_pi(heading_robot-heading_person)
        orientation_heading = np.rad2deg(heading_relative)
        #print (f"ob: {gazeboros_env.get_observation()}")
        print (f"reward: {gazeboros_env.get_reward()}")
        print (f"pos: {rel_person} vs {relative_pos2}")
        print (f"orientation_h: {orientation_heading} dist: {distance} orin: {orientation1}")
        print (f"orientation_robo: {np.rad2deg(heading_robot)} orintation pers: {np.rad2deg(heading_person)}")
        print ("\n\n")

        #if step % 50==0:
        #    print("reseting")
        #    gazeboros_env.reset()

        #gazeboros_env.visualize_observation()
        rospy.sleep(1)


#test()
