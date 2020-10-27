from gym.utils import seeding
from datetime import datetime

import copy
import os, subprocess, time, signal

import gym

import math
import random
import _thread
# u
import numpy as np
import cv2 as cv

import rospy

from squaternion import quat2euler
from squaternion import euler2quat

import actionlib
# Brings in the .action file and messages used by the move base action
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Point32
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock

from costmap_converter.msg import ObstacleArrayMsg
from costmap_converter.msg import ObstacleMsg

from simple_pid import PID

import pickle

import threading

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
    def __init__(self, name, max_angular_speed=1, max_linear_speed=1, relative=None, use_goal=False, use_movebase=False):
        self.name = name
        self.init_node = False
        self.alive = True
        self.prev_call_vicon_ = None
        if relative is None:
            relative = self
        self.relative = relative
        self.log_history = []
        self.init_node = True
        self.deleted = False
        self.update_rate_states = 2.0
        self.use_movebase = use_movebase
        self.window_size_history = 10
        self.use_goal = use_goal
        self.current_vel_ = Twist()
        self.collision_distance = 0.5
        self.max_angular_vel = max_angular_speed
        self.max_linear_vel = max_linear_speed
        self.max_rel_pos_range = 5.0 # meter
        self.max_laser_range = 5.0 # meter
        self.width_laserelement_image = 100
        self.height_laser_image = 50
        self.state_ = {'position':      (None, None),
                       'orientation':   None}
        self.action_client_ = None

        self.cmd_vel_pub =  rospy.Publisher('/cmd_vel_agent', Twist, queue_size=1)
        self.angular_pid = PID(0.5, 0, 0.03, setpoint=0)
        self.linear_pid = PID(1.5, 0, 0.05, setpoint=0)
        self.pos_history = History(self.window_size_history, self.update_rate_states)
        self.orientation_history = History(self.window_size_history, self.update_rate_states)
        self.velocity_history = History(self.window_size_history, self.update_rate_states)
        self.is_collided = False
        self.is_pause = False
        self.reset = False
        self.scan_image = None
        if self.name == "robot":
            rospy.Subscriber("/vicon/Robot/Robot", TransformStamped, self.vicon_cb)
            # Create an action client called "move_base" with action definition file "MoveBaseAction"
            self.action_client_ = actionlib.SimpleActionClient('/move_base_0', MoveBaseAction)
            # Waits until the action server has started up and started listening for goals.
            rospy.loginfo("wait for service")
            self.action_client_.wait_for_server(rospy.rostime.Duration(0.4))
            rospy.loginfo("wait for service")
        elif self.name == "person":
            self.obstacle_pub_ =  rospy.Publisher('/move_base_node_0/TebLocalPlannerROS/obstacles', ObstacleArrayMsg, queue_size=1)
            rospy.Subscriber("/vicon/Person/Person", TransformStamped, self.vicon_cb)
        else:
            rospy.logerr("wrong name {}".format(self.name))
            exit(10)


    
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
        move_base_goal.target_pose.header.frame_id = "world".format(0)
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

    def update(self):

        self.alive = True
        self.goal = {"pos": None, "orientation": None}
        self.angular_pid = PID(2.5, 0, 0.03, setpoint=0)
        self.linear_pid = PID(2.5, 0, 0.05, setpoint=0)
        self.log_history = []
        #self.prev_call_vicon_ = None
        #self.is_collided = False
        self.is_pause = False
        self.reset = False
    def add_log(self, log):
        self.log_history.append(log)

    def remove(self):
        self.reset = True

    def vicon_cb(self, pose_msg):
        # if self.prev_call_vicon_ is not None and rospy.Time.now().to_sec() - self.prev_call_vicon_ < 0.05:
        #     return
        pos = pose_msg.transform.translation
        prev_state = copy.copy(self.state_)
        self.state_["position"] = (pos.x, pos.y)
        euler = quat2euler(pose_msg.transform.rotation.x, pose_msg.transform.rotation.y, pose_msg.transform.rotation.z, pose_msg.transform.rotation.w)
        self.state_["orientation"] = euler[0]
        self.add_log((self.state_["position"][0],self.state_["position"][1], euler[0]))
        if self.prev_call_vicon_ is None:
            self.prev_call_vicon_ = rospy.Time.now().to_sec()
            return
        if abs(rospy.Time.now().to_sec() - self.prev_call_vicon_) < 0.1:
          return

        # if self.relative is not None and not self.relative.reset:
        #     orientation_rel, position_rel = ViconEnv.get_relative_heading_position(self, self.relative)
        #     if orientation_rel is None or position_rel is None:
        #         ropy.logwarn("por or orientation is None")
        #     else:
        #         self.relative_orientation_history.add_element(orientation_rel, rospy.Time.now().to_sec())
        #         self.relative_pos_history.add_element(position_rel, rospy.Time.now().to_sec())

        # get velocity
        twist = Twist()
        delta_time = rospy.Time.now().to_sec() - self.prev_call_vicon_
        twist.linear.x = (prev_state["position"][0] - self.state_["position"][0]) / delta_time 
        twist.angular.z = ViconEnv.wrap_pi_to_pi(prev_state["orientation"]-self.state_["orientation"])/delta_time
        self.prev_call_vicon_ = rospy.Time.now().to_sec()
        self.velocity_history.add_element(np.asarray((twist.linear.x, twist.angular.z)))
        if self.name == "robot":
          position_rel = ViconEnv.get_relative_position((pos.x, pos.y), self.relative)
          orientation_rel = ViconEnv.wrap_pi_to_pi(self.state_["orientation"]-self.relative.state_["orientation"])
          if self.relative is not None:
              rospy.loginfo("{}: rel_pos: {:2.2f},{:2.2f} orientation {:2.2f} vel_robot: {:2.2f} o:{:2.2f} vel_person {:2.2f} o:{:2.2f}".format(self.name, position_rel[0], position_rel[1], np.rad2deg(orientation_rel), self.get_velocity()[0],np.rad2deg(self.get_velocity()[1]),self.relative.get_velocity()[0],np.rad2deg(self.relative.get_velocity()[1]) ))
        
        self.orientation_history.add_element(self.state_["orientation"])
        self.pos_history.add_element(self.state_["position"])
        if self.name == "person":
            obstacle_msg_array = ObstacleArrayMsg()
            obstacle_msg_array.header.stamp = rospy.Time.now()
            obstacle_msg_array.header.frame_id = "world"
            obstacle_msg = ObstacleMsg()
            obstacle_msg.header = obstacle_msg_array.header
            obstacle_msg.id = 0
            for x in range (5):
                for y in range (5):
                    point = Point32()
                    point.x = pos.x + (x-2)*0.1
                    point.y = pos.y + (y-2)*0.1
                    point.z = 0.2
                    obstacle_msg.polygon.points.append(point)
            obstacle_msg.orientation.x = pose_msg.transform.rotation.x
            obstacle_msg.orientation.y = pose_msg.transform.rotation.y
            obstacle_msg.orientation.z = pose_msg.transform.rotation.z
            obstacle_msg.orientation.w = pose_msg.transform.rotation.w
            obstacle_msg.velocities.twist.linear.x = twist.linear.x
            obstacle_msg.velocities.twist.angular.z = twist.linear.z
            obstacle_msg_array.obstacles.append(obstacle_msg)
            self.obstacle_pub_.publish(obstacle_msg_array)

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
            pos = ViconEnv.denormalize(action[0:2], self.max_rel_pos_range)
            pos_global = ViconEnv.get_global_position(pos, self.relative)
            self.goal["orientation"] = self.get_orientation()
            self.goal["pos"] = pos_global
            if self.use_movebase:
                #orientation = ViconEnv.denormalize(action[2], math.pi)
                self.movebase_client_goal(pos_global, self.goal["orientation"])
        else:
            angular_vel = max(min(action[1]*self.max_angular_vel, self.max_angular_vel), -self.max_angular_vel)
            linear_vel = max(min(action[0]*self.max_linear_vel, self.max_linear_vel), -self.max_linear_vel)
            #if (abs(angular_vel)>1 and abs(linear_vel)>0.4):
            #  linear_vel *= 0.6

            cmd_vel = Twist()
            cmd_vel.linear.x = linear_vel#float(self.current_vel_.linear.x -(self.current_vel_.linear.x - linear_vel)*0.9)
            cmd_vel.angular.z = -angular_vel#-float(self.current_vel_.angular.z - (self.current_vel_.angular.z - angular_vel)*0.9)
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

    def go_to_pos(self, pos, stop_after_getting=False, person_mode=0):
        print ("pause {} reset {}".format(self.is_pause, self.reset))
        if self.is_pause:
            self.stop_robot()
            return
        if self.reset:
            return
        diff_angle, distance = self.angle_distance_to_point(pos)
        print("dist: {}angle: {}".format(distance, np.rad2deg(diff_angle)))
        time_prev = rospy.Time.now().to_sec()
        while (not distance < 0.2 and abs(rospy.Time.now().to_sec() - time_prev) < 5) or person_mode>0  :
            if self.is_pause:
                self.stop_robot()
                return
            if self.reset:
                return
            diff_angle, distance = self.angle_distance_to_point(pos)
            if distance is None:
                return

            angular_vel = -min(max(self.angular_pid(diff_angle)*10, -self.max_angular_vel),self.max_angular_vel)
            linear_vel = min(max(self.linear_pid(-distance), -self.max_linear_vel), self.max_linear_vel)
            linear_vel = min(linear_vel * math.pow((abs(math.pi - abs(diff_angle))/math.pi), 2), linear_vel)
            print("angle: {} multiplier {} distance: {}".format(math.pi - diff_angle, abs(math.pi - abs(diff_angle))/math.pi, distance))

            if self.reset:
                return
            cmd_vel = Twist()
            if person_mode == 1:
                linear_vel = 0.4
                angular_vel = 0.4
            elif person_mode == 2:
                linear_vel = 0.4
                angular_vel = -0.4
            elif person_mode == 3:
                linear_vel = 0.2
                angular_vel = -0.4
            elif person_mode == 4:
                linear_vel = 0.2
                angular_vel = 0.4
            elif person_mode == 5:
                linear_vel = 0.5
                angular_vel = 0.0
            elif person_mode == 6:
                linear_vel, angular_vel = self.get_velocity()[0]
                linear_vel = linear_vel - (linear_vel - (random.random()/2 + 0.5))/2.
                angular_vel = angular_vel - (angular_vel - (random.random()-0.5)*2)/2.
            elif person_mode == 7:
                linear_vel = 0.5
                angular_vel = -0.5


            cmd_vel.linear.x = float(linear_vel)
            cmd_vel.angular.z = float(angular_vel)
            self.cmd_vel_pub.publish(cmd_vel)
            time.sleep(0.01)

        if stop_after_getting:
            self.stop_robot()

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

class ViconEnv(gym.Env):

    def __init__(self, is_evaluation=False):

        self.node = rospy.init_node('gym_vicon')
        self.is_evaluation_ = is_evaluation

        self.is_reseting = True
        self.lock = _thread.allocate_lock()
        self.robot_mode = 0
        self.use_goal = True
        self.use_movebase = True
        self.is_use_test_setting = False

        self.center_pos_ = (0, 0)
        self.current_obsevation_image_ = np.zeros([2000,2000,3])
        self.current_obsevation_image_.fill(255)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(47,))
        self.prev_action = (0, 0)

        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.min_distance = 1
        self.max_distance = 2.5
        if self.is_evaluation_:
           self.max_numb_steps = 1000000000000000000
        else:
            self.max_numb_steps = 2000000000000000
        self.reward_range = [-1, 1]
    

    def is_skip_run(self):
        return False
    
    def get_angle_person_robot(self):
        _, pos_rel = ViconEnv.get_relative_heading_position(self.robot, self.person)
        angle_robot_person = math.atan2(pos_rel[1], pos_rel[0])
        return (ViconEnv.wrap_pi_to_pi(angle_robot_person))
    
    def use_test_setting(self):
        self.is_use_test_setting = True


    def set_agent(self, agent_num):
        date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.log_file = open('log_{}.pkl'.format(date_time), "wb")
        self.agent_num = agent_num
        self.create_robots()

        self.init_system()
        rospy.loginfo("in set aget")
    


    def create_robots(self):

        self.person = Robot('person',
                            max_angular_speed=1, max_linear_speed=.6)


        relative = self.person
        if self.use_goal:
          relative = self.person
        self.robot = Robot('robot',
                            max_angular_speed=1.8, max_linear_speed=0.8, relative=relative, use_goal=self.use_goal, use_movebase=self.use_movebase)

    def get_supervised_action(self):
        while not self.person.is_current_state_ready() and not self.is_reseting:
            time.sleep(0.1)
        if self.is_reseting:
            return np.asarray([0,0])

        pos = self.person.calculate_ahead(1.5)
        pos_person = self.person.get_pos()
        pos_relative = ViconEnv.get_relative_position(pos, self.robot.relative)
        pos_person_relative = ViconEnv.get_relative_position(pos_person, self.robot.relative)
        print (f"pos pos_person pos_relative [pos], [pos_person] [pos_relative]")
        pos_norm = ViconEnv.normalize(pos_relative, self.robot.max_rel_pos_range)
        orientation = ViconEnv.normalize(math.atan2(pos_relative[1] - pos_person_relative[1], pos_relative[0] - pos_person_relative[0]), math.pi)
        return np.asarray((pos_norm[0], pos_norm[1], orientation))

    def init_system(self):

        self.number_of_steps = 0
        rospy.loginfo("init called")
        self.is_reseting = False
        self.is_pause = True

        self.robot.update()
        self.person.update()
        
        self.center_pos_ = self.person.state_["position"]
        self.path_finished = False

        self.robot.reset = False
        self.person.reset = False

        # self.resume_simulator()
        rospy.loginfo("init finished")

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
            angle -= math.pi
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
            rospy.loginfo ("waiting for observation to be ready")
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
            time.sleep(0.01)
            if counter > 10000:
                rospy.loginfo( "path follower waiting for pause to be false")
                counter = 0
            counter += 1
        rospy.loginfo( "path follower waiting for lock pause:{} reset:{}".format(self.is_pause, self.is_reseting))
        if self.lock.acquire(timeout=10):
            rospy.loginfo("path follower got the lock")
            if self.is_evaluation_:
                mode_person = 0
            else:
                mode_person = random.randint(0, 8)
            for idx in range (idx_start, len(self.path["points"])-3):
                point = (self.path["points"][idx][0], self.path["points"][idx][1])
                self.current_path_idx = idx
                counter_pause = 0
                while self.is_pause:
                    counter_pause+=1
                    rospy.loginfo("pause in path follower")
                    if self.is_reseting or counter_pause > 200:
                        self.lock.release()
                        return
                    time.sleep(0.1)
                try:
                    self.person.go_to_pos(point, True, mode_person)
                    # person_thread = threading.Thread(target=self.person.go_to_pos, args=(point, True, mode_person))
                    # person_thread.start()
                    # if self.robot_mode == 1:
                    #     noisy_point = (self.path["points"][idx+3][0] +min(max(np.random.normal(),-0.5),0.5), self.path["points"][idx+3][1] +min(max(np.random.normal(),-0.5),0.5))

                    #     robot_thread = threading.Thread(target=self.robot.go_to_pos, args=(noisy_point,True,))
                    #     robot_thread.start()
                    #     robot_thread.join()

                    # person_thread.join()

                except Exception as e:
                    rospy.logwarn("path follower {}, {}".format(self.is_reseting, e))
                    break
                rospy.logdebug("got to point: {} out of {}".format(idx - idx_start, len(self.path["points"]) - idx_start ))
                if self.is_reseting:
                    self.person.stop_robot()
                    break
            self.lock.release()
            rospy.loginfo("path follower release the lock")
            self.path_finished = True
        else:
            rospy.logerr("problem in getting the log in path follower")
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
        rospy.loginfo("in get observation")
        while self.robot.pos_history.avg_frame_rate is None or self.robot.velocity_history.avg_frame_rate is None or self.person.velocity_history.avg_frame_rate is None:
            if self.is_reseting:
                rospy.loginfo("reseting return none")
                return None
            time.sleep(0.1)
            rospy.loginfo("waiting to get pos/vel pos: {} vel: {} vel_person: {}".format(self.robot.pos_history.avg_frame_rate ,self.robot.velocity_history.avg_frame_rate, self.person.velocity_history.avg_frame_rate))
        # pose_history = np.asarray(self.robot.relative_pos_history.get_elemets()).flatten()/5.0
        # heading_history = np.asarray(self.robot.relative_orientation_history.get_elemets())/math.pi
        # print("pos: {:2.2f} {:2.2f} orientation:{:2.2f} vel l,a: {:2.2f} {:2.2f}".format(self.robot.relative_pos_history.get_elemets()[0][0], self.robot.relative_pos_history.get_elemets()[0][1], np.rad2deg(self.robot.relative_orientation_history.get_elemets()[0]), self.robot.get_velocity()[0], np.rad2deg(self.robot.get_velocity()[1])))
        # orientation_position = np.append(pose_history, heading_history)
        # velocities = np.concatenate((self.person.get_velocity(), self.robot.get_velocity()))
        # return np.append(orientation_position, velocities)
        
        pos_his_robot = np.asarray(self.robot.pos_history.get_elemets())
        heading_robot = self.robot.state_["orientation"]

        pos_his_person = np.asarray(self.person.pos_history.get_elemets())
        heading_person = self.person.state_["orientation"]

        robot_vel = np.asarray(self.robot.get_velocity())
        person_vel = np.asarray(self.person.get_velocity())
        poses = np.concatenate((pos_his_robot, pos_his_person))

        
        heading_relative = ViconEnv.wrap_pi_to_pi(heading_robot-heading_person)/(math.pi)
        pos_rel = []
        for pos in (poses):
            relative = ViconEnv.get_relative_position(pos, self.robot.relative)
            pos_rel.append(relative)
        pos_history = np.asarray(np.asarray(pos_rel)).flatten()/6.0
        #TODO: make the velocity normalization better
        velocities = np.concatenate((person_vel, robot_vel))/self.robot.max_angular_vel
        velocities_heading = np.append(velocities, heading_relative)
        final_ob =  np.append(np.append(pos_history, velocities_heading), self.prev_action)

        return final_ob


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

    @staticmethod
    def get_relative_heading_position(relative, center):
        while not relative.is_current_state_ready() or not center.is_current_state_ready():
            if relative.reset:
                rospy.loginfo("reseting so return none in rel pos rel: {} center".format(relative.is_current_state_ready(), center.is_current_state_ready()))
                return (None, None)
            time.sleep(0.01)
            rospy.loginfo("waiting for observation to be ready heading position")
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
        return angle_relative, relative_pos


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

        heading_relative = ViconEnv.wrap_pi_to_pi(heading_robot-heading_person)/(math.pi)
        center_pos = pos_his_robot[-1]
        for pos in pos_his_robot:
            relative = ViconEnv.get_relative_position(pos, self.robot)
            pos_rel = ViconEnv.to_image_coordinate(relative, (0, 0))
            pos_gt = ViconEnv.to_image_coordinate(pos, center_pos)
            observation_image = self.add_circle_observation_to_image(relative, (255, 0, 0), 10, center_pos=(0,0), image=observation_image)
            observation_image_gt = self.add_circle_observation_to_image(pos, (255, 0, 0), 10, center_pos=center_pos, image=observation_image_gt)

        for pos in pos_his_person:
            relative = ViconEnv.get_relative_position(pos, self.robot)
            pos_rel = ViconEnv.to_image_coordinate(relative, (0, 0))
            pos_gt = ViconEnv.to_image_coordinate(pos, center_pos)
            observation_image = self.add_circle_observation_to_image(relative, (0, 255, 0), 10, image = observation_image, center_pos=(0,0))
            observation_image_gt = self.add_circle_observation_to_image(pos, (0, 255, 0), 10, image=observation_image_gt, center_pos=center_pos)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(observation_image, encoding="bgr8"))
        self.image_pub_gt.publish(self.bridge.cv2_to_imgmsg(observation_image_gt, encoding="bgr8"))
    
    def get_current_observation_image(self):

        image = self.current_obsevation_image_
        image = image/255.

        return image


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
            self.add_line_observation_to_image(robot_pos, current_goal["pos"])
        else:
            self.add_line_observation_to_image(robot_pos, person_pos)
        alpha = 0.50
        self.current_obsevation_image_ = cv.addWeighted(self.new_obsevation_image_, alpha, self.current_obsevation_image_, 1 - alpha, 0)


    @staticmethod
    def to_image_coordinate(pos, center_pos):
        return (int((pos[0] - center_pos[0])*50+1000), int((pos[1] - center_pos[1])*50+1000))

    def add_line_observation_to_image(self, pos, pos2):
        color = self.colors_visualization[self.color_index]
        pos_image = ViconEnv.to_image_coordinate(pos, self.center_pos_)
        pos_image2 = ViconEnv.to_image_coordinate(pos2, self.center_pos_)
        if pos_image[0] >2000 or pos_image[0] < 0 or pos_image[1] >2000 or pos_image[1] < 0:
            rospy.logerr("problem with observation: {}".format(pos_image))
            return
        self.new_obsevation_image_ = cv.line(self.new_obsevation_image_, (pos_image[0], pos_image[1]), (pos_image2[0], pos_image2[1]), color, 1)

    def add_triangle_observation_to_image(self, pos, orientation):
        color = self.colors_visualization[self.color_index]
        pos_image = ViconEnv.to_image_coordinate(pos, self.center_pos_)
        pos_triangle1 = ViconEnv.to_image_coordinate((pos[0]+math.cos(orientation)*0.3, pos[1]+math.sin(orientation)*0.3), self.center_pos_)
        pos_triangle2 = ViconEnv.to_image_coordinate((pos[0]+math.cos(orientation+math.pi/2)*0.1, pos[1]+math.sin(orientation+math.pi/2)*0.1), self.center_pos_)
        pos_triangle3 = ViconEnv.to_image_coordinate((pos[0]+math.cos(orientation-math.pi/2)*0.1, pos[1]+math.sin(orientation-math.pi/2)*0.1), self.center_pos_)
        poses = [pos_triangle1, pos_triangle2, pos_triangle3]
        print(poses)

        for pos in poses:
            if pos[0] >2000 or pos[0] < 0 or pos[1] >2000 or pos[1] < 0:
                rospy.logerr("problem with observation: {}".format(pos))
                return
        self.new_obsevation_image_ = cv.drawContours(self.new_obsevation_image_, [np.asarray(poses)], 0, color, -1)


    def add_arrow_observation_to_image(self, pos, orientation):
        color = self.colors_visualization[self.color_index]
        pos_image = ViconEnv.to_image_coordinate(pos, self.center_pos_)
        pos_image2 = ViconEnv.to_image_coordinate((pos[0]+math.cos(orientation)*0.3, pos[1]+math.sin(orientation)*0.3), self.center_pos_)
        if pos_image[0] >2000 or pos_image[0] < 0 or pos_image[1] >2000 or pos_image[1] < 0:
            rospy.logerr("problem with observation: {}".format(pos_image))
            return
        self.new_obsevation_image_ = cv.arrowedLine(self.new_obsevation_image_, (pos_image[0], pos_image[1]), (pos_image2[0], pos_image2[1]), color, 2, tipLength=0.5)

    def add_circle_observation_to_image(self, pos, color, radious, center_pos=None, image=None):
        if image is None:
            image = self.new_obsevation_image_
        if center_pos is None:
            center_pos = self.center_pos_
        pos_image = ViconEnv.to_image_coordinate(pos, center_pos)
        if pos_image[0] >2000 or pos_image[0] < 0 or pos_image[1] >2000 or pos_image[1] < 0:
            rospy.logerr("problem with observation: {}".format(pos_image))
            return
        return (cv.circle(image , (pos_image[0], pos_image[1]), radious, color, 2))

    def __del__(self):
        # todo
        return

    def take_action(self, action):
        self.prev_action = action[:2]
        self.robot.take_action(action)
        return

    def step(self, action):
        try:
          self.number_of_steps += 1
          self.take_action(action)
          rospy.sleep(0.1)
          reward = self.get_reward()
          ob = self.get_observation()
          episode_over = False
          rel_person = ViconEnv.get_relative_heading_position(self.robot, self.person)[1]

          distance = math.hypot(rel_person[0], rel_person[1])
          if self.path_finished:
              rospy.loginfo("path finished")
              episode_over = True
          if self.is_collided():
              episode_over = True
              rospy.loginfo('collision happened episode over')
              reward -= 0.5
          elif distance > 5:
              episode_over = True
              rospy.loginfo('max distance happened episode over')
          elif self.number_of_steps > self.max_numb_steps:
              episode_over = True
              rospy.loginfo('max number of steps episode over')
          reward = min(max(reward, -1), 1)
          rospy.loginfo("action {} reward {}".format(action, reward))
          #reward += 1
        except Exception as e:
          print (e)
          exit(19)
        episode_over = False
        return ob, reward, episode_over, {}

    def is_collided(self):
        rel_person = ViconEnv.get_relative_heading_position(self.robot, self.person)[1]
        distance = math.hypot(rel_person[0], rel_person[1])
        if distance < 0.8 or self.robot.is_collided:
            return True
        return False

    def get_reward(self):
        reward = 0
        angle_robot_person, pos_rel = ViconEnv.get_relative_heading_position(self.robot, self.person)
        angle_robot_person = math.atan2(pos_rel[1], pos_rel[0])
        angle_robot_person = np.rad2deg(angle_robot_person)
        distance = math.hypot(pos_rel[0], pos_rel[1])
        # Negative reward for being behind the person
        if self.is_collided():
            reward -= 1
        if distance < 0.3:
            reward = -1.3
        elif abs(distance - 1.7) < 0.7:
            reward += 0.1 * (0.7 - abs(distance - 1.7))
        elif distance >= 1.7:
            reward -= 0.25 * (distance - 1.7)
        elif distance < 1:
            reward -= (1 - distance)/2.0
        if abs(angle_robot_person) < 45:
            reward += 0.2 * (45 - abs(angle_robot_person)) / 45
        else:
            reward -= 0.1 * abs(angle_robot_person) / 180

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
                self.init_system()
                not_init = False
            except RuntimeError as e:
                rospy.logerr("error happend reseting: {}".format(e))
        if not_init:
            rospy.loginfo("not init so run reset again")
            return (self.reset())
        else:
            return self.get_observation()

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        return


def read_bag():
    vicon_n = ViconEnv()
    vicon_n.set_agent(0)

    while vicon_n.robot.prev_call_vicon_ is None or rospy.Time.now().to_sec() - vicon_n.robot.prev_call_vicon_ < 5:
        rospy.sleep(0.1)
    vicon_n.save_log()
    print("done")

# read_bag()
