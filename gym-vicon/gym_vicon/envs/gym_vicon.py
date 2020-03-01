from gym.utils import seeding
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



from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock

from simple_pid import PID

import pickle

import threading

import logging

logger = logging.getLogger(__name__)

class History():
    def __init__(self, memory_size, window_size, rate):
        self.data = [None for x in range(memory_size)]
        self.idx = 0
        self.update_rate = rate
        self.memory_size = memory_size
        self.window_size = window_size
        self.avg_frame_rate = None
        self.time_data_= []

    def add_element(self, element, time_data):
        """
        element: the data that we put inside the history data array
        time_data: in second the arrive time of data (get it from ros msg time)
        """
        self.idx = (self.idx + 1) % self.window_size

        if self.data[self.idx] is None:
            for idx in range(self.memory_size):
                self.data[idx] = element
        self.data[self.idx] = element
        if not len(self.time_data_) > 50:
            self.time_data_.append(time_data)
            if len(self.time_data_) > 3:
                prev_t = self.time_data_[0]
                time_intervals = []
                for t in self.time_data_[1:]:
                    time_intervals.append(t - prev_t)
                    prev_t = t
                self.avg_frame_rate = 1.0 / np.average(time_intervals)


    def get_elemets(self):
        return_data = []
        skip_frames = -int(math.ceil(self.avg_frame_rate / self.update_rate))
        # print("in get element", skip_frames, self.avg_frame_rate, self.update_rate)
        index = (self.idx - 1)% self.window_size
        if self.window_size * abs(skip_frames) >= self.memory_size:
            print("error in get element memory not enough")
        for i in range (self.window_size):
            return_data.append(self.data[index])
            index = (index + skip_frames) % self.window_size

        return return_data

class Robot():
    def __init__(self, name, max_angular_speed=1, max_linear_speed=1, relative=None):
        self.name = name
        self.init_node = False
        self.alive = True
        self.prev_call_vicon_ = None
        self.relative = relative
        self.log_history = []
        self.init_node = True
        self.deleted = False
        self.collision_distance = 0.5
        self.max_angular_vel = max_angular_speed
        self.max_linear_vel = max_linear_speed
        self.max_laser_range = 5.0 # meter
        self.width_laserelement_image = 100
        self.height_laser_image = 50
        self.state_ = {'position':      (None, None),
                       'orientation':   None}

        self.cmd_vel_pub =  rospy.Publisher('/cmd_vel_agent', Twist, queue_size=1)
        if self.name == "robot":
            rospy.Subscriber("/vicon/turttle/turttle", TransformStamped, self.vicon_cb)
        elif self.name == "person":
            rospy.Subscriber("/vicon/Personm/Personm", TransformStamped, self.vicon_cb)
        else:
            rospy.logerr("wrong name {}".format(self.name))
            exit(10)


        self.angular_pid = PID(0.5, 0, 0.03, setpoint=0)
        self.linear_pid = PID(1.5, 0, 0.05, setpoint=0)
        self.relative_pos_history = History(200, 10, 2)
        self.relative_orientation_history = History(200, 10, 2)
        self.velocity_history = History(200, 10, 2)
        self.is_collided = False
        self.is_pause = False
        self.reset = False
        self.scan_image = None


    def is_current_state_ready(self):
        return (self.state_['position'][0] is not None)

    def is_observation_ready(self):
        return (self.relative_pos_history.avg_frame_rate is not None and\
                self.relative_orientation_history.avg_frame_rate is not None and\
                self.velocity_history.avg_frame_rate is not None)

    def update(self):

        self.alive = True
        self.angular_pid = PID(0.5, 0, 0.03, setpoint=0)
        self.linear_pid = PID(1.5, 0, 0.05, setpoint=0)
        #self.relative_pos_history = History(200, 10, 2)
        #self.relative_orientation_history = History(200, 10, 2)
        #self.velocity_history = History(200, 10, 2)
        #self.velocity_history.add_element((0,0), rospy.Time.now().to_sec())
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
        if self.prev_call_vicon_ is not None and rospy.Time.now().to_sec() - self.prev_call_vicon_ < 0.05:
            return
        pos = pose_msg.transform.translation
        prev_state = copy.copy(self.state_)
        self.state_["position"] = (pos.y, -pos.x)
        euler = quat2euler(pose_msg.transform.rotation.x, pose_msg.transform.rotation.y, pose_msg.transform.rotation.z, pose_msg.transform.rotation.w)
        self.state_["orientation"] = euler[0]
        if self.prev_call_vicon_ is None:
            self.prev_call_vicon_ = rospy.Time.now().to_sec()
            return

        if self.relative is not None and not self.relative.reset:
            orientation_rel, position_rel = ViconEnv.get_relative_heading_position(self, self.relative)
            if orientation_rel is None or position_rel is None:
                ropy.logwarn("por or orientation is None")
            else:
                self.relative_orientation_history.add_element(orientation_rel, rospy.Time.now().to_sec())
                self.relative_pos_history.add_element(position_rel, rospy.Time.now().to_sec())

        # get velocity
        twist = Twist()
        delta_time = rospy.Time.now().to_sec() - self.prev_call_vicon_
        twist.linear.x = math.hypot(prev_state["position"][0] - self.state_["position"][0], prev_state["position"][1] - self.state_["position"][1]) / delta_time 
        twist.angular.z = ViconEnv.wrap_pi_to_pi(prev_state["orientation"]-self.state_["orientation"])/delta_time
        self.prev_call_vicon_ = rospy.Time.now().to_sec()
        self.velocity_history.add_element(np.asarray((twist.linear.x, twist.angular.z)), rospy.Time.now().to_sec())
        if self.relative is not None:
            # rospy.loginfo("{}: delta {} rel_pos: {} orientation {} vel ang: {} linea: {}".format(self.name, delta_time, position_rel, np.rad2deg(orientation_rel), np.rad2deg(twist.angular.z), twist.linear.x))
            rospy.loginfo("{}: rel_pos: {:2.2f},{:2.2f} orientation {:2.2f}".format(self.name, position_rel[0], position_rel[1], np.rad2deg(orientation_rel)))
        # else:
        #     rospy.loginfo("{}: delta {} pos: {} orientation {} vel ang: {} linea: {}".format(self.name, delta_time, self.state_["position"], np.rad2deg(self.state_["orientation"]), np.rad2deg(twist.angular.z), twist.linear.x))

    def get_velocity(self):
        return self.velocity_history.get_elemets()

    def pause(self):
        self.is_pause = True
        self.stop_robot()

    def resume(self):
        self.is_pause = False

    # action is two digit number the first one is linear_vel (out of 9) second one is angular_vel (our of 6)
    def take_action(self, action):
        if self.is_pause:
            return

        linear_vel = max(min((1+action[0])/2., self.max_linear_vel), 0)
        angular_vel = max(min(action[1], self.max_angular_vel), -self.max_angular_vel)

        cmd_vel = Twist()
        cmd_vel.linear.x = float(linear_vel)
        cmd_vel.angular.z = float(-angular_vel)
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
        while self.scan_image is None:
            time.sleep(0.1)
        return np.expand_dims(self.scan_image, axis=2)

class ViconEnv(gym.Env):

    def __init__(self, is_evaluation=False):

        self.node = rospy.init_node('gym_vicon')
        self.is_evaluation_ = is_evaluation

        self.is_reseting = True
        self.lock = _thread.allocate_lock()
        self.robot_mode = 0

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(70,))

        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.min_distance = 1
        self.max_distance = 2.5
        if self.is_evaluation_:
           self.max_numb_steps = 1000000000000000000
        else:
            self.max_numb_steps = 2000000000000000
        self.reward_range = [-1, 1]

    def set_agent(self, agent_num):

        self.log_file = None
        self.agent_num = agent_num
        self.create_robots()

        self.init_simulator()
    


    def create_robots(self):

        self.person = Robot('person',
                            max_angular_speed=0.25, max_linear_speed=.25)

        self.robot = Robot('robot',
                            max_angular_speed=2.0, max_linear_speed=0.6, relative=self.person)

    def find_random_point_in_circle(self, radious, min_distance, around_point):
        max_r = 2
        r = (radious - min_distance) * math.sqrt(random.random()) + min_distance
        theta = random.random() * 2 * math.pi
        x = around_point[0] + r * math.cos(theta)
        y = around_point[1] + r * math.sin(theta)
        return (x, y)


    def init_simulator(self):

        self.number_of_steps = 0
        rospy.loginfo("init simulation called")
        self.is_reseting = False
        self.is_pause = True

        self.robot.update()
        self.person.update()

        self.path_finished = False

        self.robot.reset = False
        self.person.reset = False

        # self.resume_simulator()
        rospy.loginfo("init simulation finished")

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
        relative_pos = (-relative_pos[1], -relative_pos[0])
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
        while self.robot.relative_pos_history.avg_frame_rate is None or self.robot.velocity_history.avg_frame_rate is None or self.person.velocity_history.avg_frame_rate is None:
            if self.is_reseting:
                return None
            time.sleep(0.1)
            rospy.logerr("waiting to get pos/vel pos: {} vel: {} vel_person: {}".format(self.robot.relative_pos_history.avg_frame_rate ,self.robot.velocity_history.avg_frame_rate, self.person.velocity_history.avg_frame_rate))
        pose_history = np.asarray(self.robot.relative_pos_history.get_elemets()).flatten()/5.0
        heading_history = np.asarray(self.robot.relative_orientation_history.get_elemets())/math.pi
        print("pos: {:2.2f} {:2.2f} orientation:{:2.2f} vel l,a: {:2.2f} {:2.2f}".format(self.robot.relative_pos_history.get_elemets()[0][0], self.robot.relative_pos_history.get_elemets()[0][1], np.rad2deg(self.robot.relative_orientation_history.get_elemets()[0]), self.robot.get_velocity()[0][0], np.rad2deg(self.robot.get_velocity()[0][1])))
        # self.visualize_observation(poses, headings, self.get_laser_scan())
        orientation_position = np.append(pose_history, heading_history)
        velocities = np.concatenate((self.person.get_velocity(), self.robot.get_velocity()))
        return np.append(orientation_position, velocities)

    def __del__(self):
        # todo
        return

    def take_action(self, action):
        self.robot.take_action(action)
        return

    def step(self, action):
        self.number_of_steps += 1
        self.take_action(action)
        time.sleep(0.15)
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
            return self.get_observation()

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        return


# def test():
#     vicon_n = ViconEnv()
#     vicon_n.set_agent(0)
#     rospy.spin()
#     print("done")
# 
# test()
