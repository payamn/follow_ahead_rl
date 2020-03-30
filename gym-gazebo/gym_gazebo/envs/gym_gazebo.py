from gym.utils import seeding


import os, subprocess, time, signal

import gym

import math
import random
import _thread
# u
import numpy as np
import cv2 as cv

try:
    import rospy
    ROS_VERSION=1
    print ("using ros1")
except Exception as e:
    import rclpy
    print ("using ros2")
    ROS_VERSION=2

from squaternion import quat2euler
from squaternion import euler2quat

from ament_index_python.packages import get_package_share_directory

from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock

from simple_pid import PID

import pickle

import threading

import logging

logger = logging.getLogger(__name__)

class Manager:
    def __init__(self, agent_num):
        self.lock_spin = _thread.allocate_lock()
        self.agent_num = agent_num
        # self.sdf_file_path = os.path.join(
        #     get_package_share_directory("turtlebot3_gazebo"), "models",
        #     "turtlebot3_burger", "model.sdf")
        self.sdf_file_path = os.path.join("/home/payam/ros2_ws/src/follow_ahead_rl/worlds/", "turtlebot3_burger", "model.sdf")
        self.node = rclpy.create_node("manager_{}".format(agent_num))
        self.time = self.node.get_clock()
        self.node.get_logger().debug(
            'Creating Service client_delete to connect to `/delete_entity`')
        self.client_delete = self.node.create_client(DeleteEntity, "/delete_entity")

        self.node.get_logger().debug(
            'Creating Service client_spawn to connect to `/spawn_entity`')
        self.client_spawn = self.node.create_client(SpawnEntity, "/spawn_entity")
        self.set_entity_state = self.node.create_client(SetEntityState, "/set_entity_state")
        self.client_pause = self.node.create_client(Empty, "/pause_physics")
        self.client_unpause = self.node.create_client(Empty, "/unpause_physics")
        self.current_time = None

    def cleanup(self):
        self.node.destroy_client(self.client_delete )
        self.node.destroy_client(self.client_spawn)
        self.node.destroy_client(self.client_pause)
        self.node.destroy_client(self.client_unpause)
        self.node.destroy_node()

    def get_time_sec(self):
        #return self.time.now().nanoseconds/1000000000.0
        while self.current_time is None:
            time.sleep(0.001)
            self.node.get_logger().debug("waiting for time")
        return self.current_time
    def remove_object(self, name):
        # self.pause()
        if not self.client_delete.service_is_ready():
            self.client_delete.wait_for_service(timeout_sec=8)
            self.node.get_logger().debug("...connected!")
        # Set data for request
        request = DeleteEntity.Request()
        request.name = name
        future = self.client_delete.call_async(request)
        # self.unpause()
        with self.lock_spin:
            self.node.get_logger().info("Sending service request to `/delete_entity`")
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=5)
        if future.result() is None:
            raise RuntimeError(
                'exception while calling service: %r' % future.exception())

    def move_robot(self, name, name_space, pos_rotation):
        translation, rotation= pos_rotation["pos"], pos_rotation["orientation"]
        # self.remove_object(name)
        # self.node.get_logger().info("Connecting to `/set_entity_state` service...")
        if not self.set_entity_state.service_is_ready():
            self.set_entity_state.wait_for_service(timeout_sec=8)
            self.node.get_logger().debug("set entity state connected!")

        quaternion_rotation = euler2quat(0, rotation, 0)

        # Set data for request
        request = SetEntityState.Request()
        request.state.name = name
        # Set position
        request.state.pose.position.x = float(translation[0])
        request.state.pose.position.y = float(translation[1])
        request.state.pose.position.z = self.agent_num * 2.5 + 0.19
        # Set orientation
        request.state.pose.orientation.x = quaternion_rotation[3]
        request.state.pose.orientation.y = quaternion_rotation[1]
        request.state.pose.orientation.z = quaternion_rotation[2]
        request.state.pose.orientation.w = quaternion_rotation[0]

        future = self.set_entity_state.call_async(request)
        # self.node.get_logger().info("before lock `/set_entity_state`")

        with self.lock_spin:
            # self.node.get_logger().info("Sending service request to `/set_entity_state`")
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=13)
        if future.result() is None:
            raise RuntimeError(
                'exception while calling set_entity state: {} name: {}'.format( future.exception(), name))

    def create_robot(self, name, name_space, pos_rotation):
        translation, rotation= pos_rotation["pos"], pos_rotation["orientation"]
        # self.remove_object(name)
        # self.node.get_logger().info("Connecting to `/spawn_entity` service...")
        if not self.client_spawn.service_is_ready():
            self.client_spawn.wait_for_service(timeout_sec=8)
            self.node.get_logger().debug("...connected!")

        # Set data for request
        request = SpawnEntity.Request()
        request.name = name
        request.xml = open(self.sdf_file_path, 'r').read()
        request.robot_namespace = name_space
        request.initial_pose.position.x = float(translation[0])
        request.initial_pose.position.y = float(translation[1])
        request.initial_pose.position.z = self.agent_num * 2.5 + 0.19
        # TODO: check euler angle
        quaternion_rotation = euler2quat(0, rotation, 0)
        request.initial_pose.orientation.x = quaternion_rotation[3]
        request.initial_pose.orientation.y = quaternion_rotation[1]
        request.initial_pose.orientation.z = quaternion_rotation[2]
        request.initial_pose.orientation.w = quaternion_rotation[0]

        future = self.client_spawn.call_async(request)
        with self.lock_spin:
            # self.node.get_logger().info("Sending service request to `/spawn_entity`")
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=13)
        if future.result() is None:
           raise RuntimeError(
                'exception while calling service: %r' % future.exception())

    def reset(self):
        pass


class History():
    def __init__(self, memory_size, window_size, rate, manager):
        self.data = [None for x in range(memory_size)]
        self.idx = 0
        self.manager = manager
        self.update_rate = rate
        self.lock = _thread.allocate_lock()
        self.memory_size = memory_size
        self.window_size = window_size
        self.avg_frame_rate = None
        self.time_data_= []

    def add_element(self, element, time_data):
        """
        element: the data that we put inside the history data array
        time_data: in second the arrive time of data (get it from ros msg time)
        """
        with self.lock:
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
        with self.lock:
            index = self.idx #(self.idx - 1)% self.window_size
            if self.window_size * skip_frames >= self.memory_size:
                print("error in get element memory not enough")
            for i in range (self.window_size):
                return_data.append(self.data[index])
                index = (index + skip_frames) % self.window_size

        return return_data

    def get_latest(self):
        with self.lock:
            return self.data[self.idx]



class Robot():
    def __init__(self, name, init_pos, manager, node, max_angular_speed=1, max_linear_speed=1, relative=None, action_mode="pos"):
        # manager.create_robot(name, name, init_pos, angle)
        manager.move_robot(name, name, init_pos)
        self.name = name
        self.manager = manager
        self.init_node = False
        self.node = rclpy.create_node(name)
        self.alive = True
        self.relative = relative
        self.log_history = []
        # self.node = node
        self.init_node = True
        self.deleted = False
        self.action_mode_ = action_mode
        self.collision_distance = 0.5
        self.max_angular_vel = max_angular_speed
        self.max_linear_vel = max_linear_speed
        self.max_laser_range = 5.0 # meter
        self.width_laserelement_image = 100
        self.height_laser_image = 50
        self.goal = None
        self.state_ = {'position':      (None, None),
                       'orientation':   init_pos["orientation"]}

        self.cmd_vel_pub = self.node.create_publisher(Twist, '/{}/cmd_vel'.format(name))
        #self.laser_sub = self.node.create_subscription(LaserScan, '/{}/scan'.format(name), self.laser_cb, qos_profile=qosProfileSensors)
        self.angular_pid = PID(0.5, 0, 0.03, setpoint=0)
        self.linear_pid = PID(1.5, 0, 0.06, setpoint=0)
        self.relative_pos_history = History(200, 10, 2, manager)
        self.relative_orientation_history = History(200, 10, 2, manager)
        self.velocity_history = History(200, 10, 2, manager)
        self.scan_image = None
       # self.scan_image_history = History(5, 1, manager)
        self.is_collided = False
        self.is_pause = False
        self.reset = False

    def update_goal(self, goal):
        self.goal = goal

    def calculate_ahead(self, distance):
        x = self.state_['position'][0] + math.cos(self.state_["orientation"]) * distance
        y = self.state_['position'][1] + math.sin(self.state_["orientation"]) * distance
        return (x,y)

    def is_current_state_ready(self):
        return (self.state_['position'][0] is not None)

    def is_observation_ready(self):
        return (self.relative_pos_history.avg_frame_rate is not None and\
                self.relative_orientation_history.avg_frame_rate is not None and\
                self.velocity_history.avg_frame_rate is not None)

    def update(self, pos_angle):
        self.reset = True

        self.manager.move_robot(self.name, self.name, pos_angle)
        self.alive = True
        self.state_ = {'position': (None, None),
                       'orientation': pos_angle["orientation"]}
        self.angular_pid = PID(0.1, 0, 0.03, setpoint=0)
        self.linear_pid = PID(1.5, 0, 0.06, setpoint=0)
        self.relative_pos_history = History(200, 10, 2, self.manager)
        self.relative_orientation_history = History(200, 10, 2, self.manager)
        self.velocity_history = History(200, 10, 2, self.manager)
        self.velocity_history.add_element((0,0), self.manager.get_time_sec())
        self.scan_image = None
        self.scan_image_history = History(5, 5, 1, self.manager)
        self.log_history = []
        self.is_collided = False
        self.is_pause = False
        self.reset = False

    def add_log(self, log):
        self.log_history.append(log)

    def spin(self):
        while self.alive:
            if self.manager.lock_spin.acquire(timeout=1):
                rclpy.spin_once(self.node)
                self.manager.lock_spin.release()
            time.sleep(0.00001)

    def remove(self):
        # self.laser_sub.destroy()
        # self.model_states_sub.destroy()
        self.reset = True

    def states_cb(self, states_msg):
        model_idx = None
        prev_pos = self.state_["position"]
        for i in range(len(states_msg.name)):
            if states_msg.name[i] == self.name:
                # self.node.get_logger().warn("statecb")
                model_idx = i
                break
        if model_idx is None:
            print("cannot find {}".format(self.name))
            return

        pos = states_msg.pose[model_idx]
        self.state_["position"] = (pos.position.x + (random.random()-0.5)/5, pos.position.y+ (random.random()-0.5)/5)
        euler = quat2euler(pos.orientation.x, pos.orientation.y, pos.orientation.z, pos.orientation.w)
        self.state_["orientation"] = euler[0] + (random.random()-0.5)*math.pi/7

        if self.relative is not None and not self.relative.reset:
            self.manager.node.get_logger().debug('before calling get rel in state cb {}'.format(self.state_))
            orientation_rel, position_rel = GazeboEnv.get_relative_heading_position(self, self.relative, self.manager.node)
            self.manager.node.get_logger().debug('after calling get rel in state cb ')
            if orientation_rel is None or position_rel is None:
                self.manager.node.get_logger().error('orientation or pos is none')
            else:
                self.manager.node.get_logger().debug("after state cb")
                self.relative_orientation_history.add_element(orientation_rel, self.manager.get_time_sec())
                self.relative_pos_history.add_element(position_rel, self.manager.get_time_sec())

        # get velocity
        twist = states_msg.twist[model_idx]
        linear_vel = twist.linear.x + + (random.random()-0.5)/5
        angular_Vel = twist.angular.z + (random.random()-0.5)*math.pi/7

        self.velocity_history.add_element(np.asanyarray((linear_vel, angular_Vel)), self.manager.get_time_sec())

    def laser_cb(self, laser_msg):

        if self.reset:
            return
        angle_increments = np.arange(float(laser_msg.angle_min) - float(laser_msg.angle_min) , float(laser_msg.angle_max) - 0.001 - float(laser_msg.angle_min), float(laser_msg.angle_increment))
        ranges = np.asarray(laser_msg.ranges, dtype=float)
        if angle_increments.shape[0] != ranges.shape[0]:
            angle_increments = angle_increments[:min(angle_increments.shape[0], ranges.shape[0])]
            ranges = ranges[:min(angle_increments.shape[0], ranges.shape[0])]
        remove_index = np.append(np.argwhere(ranges >= self.max_laser_range), np.argwhere(ranges <= laser_msg.range_min))
        angle_increments = np.delete(angle_increments, remove_index)
        ranges = np.delete(ranges, remove_index)
        min_ranges = float("inf")  if len(ranges)==0  else np.min(ranges)
        if min_ranges < self.collision_distance:
            self.is_collided = True
            self.node.get_logger().debug("robot collided:")
        x = np.floor((self.width_laser_image/2.0 - (self.width_laser_image / 2 / self.max_laser_range) * np.multiply(np.cos(angle_increments), ranges))).astype(np.int)
        y = np.floor((self.height_laser_image-1 - (self.height_laser_image / self.max_laser_range) * np.multiply(np.sin(angle_increments), ranges))).astype(np.int)
        if len(x) > 0 and (np.max(x) >= self.width_laser_image or np.max(y) >= self.height_laser_image):
            print("problem, max x:{} max y:{}".format(np.max(x), np.max(y)))
        scan_image = np.zeros((self.height_laser_image, self.width_laser_image), dtype=np.uint8)
        scan_image[y, x] = 255
        self.scan_image_history.add_element(scan_image)
        self.scan_image = scan_image

    def get_velocity(self):
        return self.velocity_history.get_elemets()

    def pause(self):
        self.is_pause = True
        self.stop_robot()

    def resume(self):
        self.is_pause = False

    # action is two digit number the first one is linear_vel (out of 9) second one is angular_vel (our of 6)
    def take_action(self, action, center_object):
        if self.is_pause:
            return
        # linear_vel = (action%10 - 4) * self.max_linear_vel / 5.0
        # angular_vel = (action//10 - 3) * self.max_angular_vel / 3.0

        if self.action_mode_ == "cmd_vel":

            linear_vel = (1+action[0])/2.*self.max_linear_vel
            linear_vel = max(min(linear_vel, self.max_linear_vel), 0)
            angular_vel = action[1]*self.max_angular_vel
            angular_vel = max(min(angular_vel, self.max_angular_vel), -self.max_angular_vel)
            # linear_vel, angular_vel = self.get_velocity()
            # linear_vel = ((1 + action[0]) / 2 * self.max_linear_vel + linear_vel) / 2
            # angular_vel = (action[1] * self.max_angular_vel + angular_vel) / 2
            cmd_vel = Twist()
            cmd_vel.linear.x = float(linear_vel)
            cmd_vel.angular.z = float(angular_vel)
            self.cmd_vel_pub.publish(cmd_vel)
        elif self.action_mode_ == "point":
            pos = GazeboEnv.get_global_position(action, center_object, self.node)
            self.update_goal(pos)

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
        while person_mode>0:
            if self.is_pause:
                self.stop_robot()
                return
            if self.reset:
                return
            if self.reset:
                return
            angular_vel = 0
            linear_vel = 0
            if person_mode == 1:
                linear_vel = self.max_linear_vel/4
                angular_vel = self.max_angular_vel/2
            elif person_mode == 2:
                linear_vel = self.max_linear_vel/4
                angular_vel = -self.max_angular_vel
            elif person_mode == 3:
                linear_vel = self.max_linear_vel/2
                angular_vel = -self.max_angular_vel/2
            elif person_mode == 4:
                linear_vel = self.max_linear_vel * random.random()
                angular_vel = 0
            elif person_mode == 5:
                linear_vel = self.max_linear_vel/4
                angular_vel = self.max_angular_vel
            elif person_mode == 6:
                linear_vel, angular_vel = self.get_velocity()[0]
                linear_vel = linear_vel - (linear_vel - (random.random()/2 + 0.5))/2.
                angular_vel = angular_vel - (angular_vel - (random.random()-0.5)*2)/2.
            elif person_mode == 7:
                linear_vel = self.max_angular_vel/3
                angular_vel = -self.max_angular_vel
            angular_vel += random.uniform(-self.max_angular_vel/4, self.max_angular_vel/4)
            linear_vel += random.uniform(self.max_linear_vel/4, self.max_linear_vel/4)
            self.publish_cmd_vel(linear_vel, angular_vel)
            time.sleep(0.002)

    def go_to_goal(self):
        while True:
            if self.reset:
                return
            while self.goal is None:
                time.sleep(0.1)
                continue
            diff_angle, distance = self.angle_distance_to_point(self.goal)
            time_prev = self.manager.get_time_sec()
            while not distance < 0.1 and abs(self.manager.get_time_sec() - time_prev) < 5:
                if self.is_pause:
                    self.stop_robot()
                    return
                if self.reset:
                    return
                diff_angle, distance = self.angle_distance_to_point(self.goal)
                if distance is None:
                    return

                if self.reset:
                    return

                angular_vel = -min(max(self.angular_pid(diff_angle)*10, -self.max_angular_vel),self.max_angular_vel)
                linear_vel = min(max(self.linear_pid(-distance), 0), self.max_linear_vel)
                linear_vel = linear_vel * math.pow((abs(math.pi - abs(diff_angle))/math.pi), 2)

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
        time_prev = self.manager.get_time_sec()
        while not distance < 0.2 and abs(self.manager.get_time_sec() - time_prev) < 5:
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

            angular_vel = -min(max(self.angular_pid(diff_angle)*10, -self.max_angular_vel),self.max_angular_vel)
            linear_vel = min(max(self.linear_pid(-distance), 0), self.max_linear_vel)
            linear_vel = linear_vel * math.pow((abs(math.pi - abs(diff_angle))/math.pi), 2)

            self.publish_cmd_vel(linear_vel, angular_vel)
            time.sleep(0.01)

        if stop_after_getting:
            self.stop_robot()

    def get_pos(self):
        counter_problem = 0
        while self.state_['position'] is None:
            if self.reset:
                return (None, None)
            if counter_problem > 20:
                self.node.get_logger().debug("waiting for pos to be available {}/{}".format(counter_problem/10, 20))
            time.sleep(0.001)
            counter_problem += 1
            if counter_problem > 200:
                raise Exception('Probable shared memory issue happend')

        return self.state_['position']

    def get_laser_image(self):
        while self.scan_image is None:
            time.sleep(0.1)
        return np.expand_dims(self.scan_image, axis=2)

class GazeboEnv(gym.Env):

    def __init__(self, is_evaluation=False):

        self.use_goal = False
        self.is_evaluation_ = is_evaluation
        self.wait_counter_ = 0

        # curriculam param
        self.max_mod_person_ = 0
        self.use_random_around_person_ = True
        self.robot_thread = None

        # being use for observation visualization
        self.center_pos_ = (0, 0)
        self.robot_color = [255,0,0]
        self.person_color = [0,0,255]
        self.goal_color = [0,255,0]

        self.action_mode_ = "point"
        self.test_simulation_ = False
        self.is_reseting = True
        self.lock = _thread.allocate_lock()
        self.robot_mode = 1
        self.current_obsevation_image_ = np.zeros([2000,2000,3])
        self.current_obsevation_image_.fill(255)

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(72,))
        # gym.spaces.Tuple(
        #     (
        #         gym.spaces.Box(low=0, high=1, shape=(50, 100, 5)),
        #         gym.spaces.Box(low=0, high=1, shape=(17,))
        #     )
        # )
        # Action space omits the Tackle/Catch actions, which are useful
        # on defense
        self.prev_action = (0,0)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.min_distance = 1
        self.max_distance = 2.5
        if self.test_simulation_ or self.is_evaluation_:
           self.max_numb_steps = 1000000000000000000
        else:
            self.max_numb_steps = 400
        self.reward_range = [-1, 1]
        self.manager = None

    def set_agent(self, agent_num):
        rclpy.init()
        self.node = rclpy.create_node("gazebo_env_{}".format(agent_num))
        # self.reset_gazebo(no_manager=True)
        self.manager = Manager(agent_num)

        self.clock_sub = self.node.create_subscription(Clock, '/clock', self.clock_cb)
        # use goal from current path to calculate reward if false use a the heading and relative position  to calculate reward
        # read the path for robot
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
        if agent_num >= 4:
            self.create_robots(create_robot_manager=True)
        else:
            self.create_robots(create_robot_manager=False)

        qosProfileSensors = rclpy.qos.qos_profile_sensor_data
        self.state_cb_prev_time = None
        self.model_states_sub = self.node.create_subscription(ModelStates, '/model_states', self.states_cb, qos_profile=qosProfileSensors)
        self.spin_thread = threading.Thread(target=self.spin, args=())
        self.spin_thread.start()
        with self.lock:
            self.init_simulator()


    def spin(self):
        while rclpy.ok():
            # while self.is_reseting:
            #     time.sleep(0.1)
            with self.manager.lock_spin:
                rclpy.spin_once(self.node)
            time.sleep(0.00001)

    def get_init_pos_robot_person(self):
        if self.is_evaluation_:
            idx_start = 0
        else:
            idx_start = random.randint(0, len(self.path["points"]) - 20)
        self.current_path_idx = idx_start

        if self.is_evaluation_:
            init_pos_person = self.path["start_person"]
            init_pos_robot = self.path["start_robot"]
        elif self.use_random_around_person_:
            init_pos_person = {"pos": self.path["points"][idx_start], "orientation": self.calculate_angle_using_path(idx_start)}
            init_pos_robot = {"pos": self.find_random_point_in_circle(3, 1, self.path["points"][idx_start]),\
                              "orientation": self.calculate_angle_using_path(idx_start)}
                              #random.random()*2*math.pi - math.pi}
        else:
            init_pos_person = {"pos": self.path["points"][idx_start], "orientation": self.calculate_angle_using_path(idx_start)}
            idx_robot = idx_start + 1
            while (math.hypot(self.path["points"][idx_robot][1] - self.path["points"][idx_start][1],
                              self.path["points"][idx_robot][0] - self.path["points"][idx_start][0]) < 1.6):
                idx_robot += 1

            init_pos_robot = {"pos": self.path["points"][idx_robot],\
                              "orientation": self.calculate_angle_using_path(idx_robot)}
        return init_pos_robot, init_pos_person


    def create_robots(self, create_robot_manager=False):
        init_pos_robot, init_pos_person = self.get_init_pos_robot_person()
        if create_robot_manager:
            self.manager.create_robot('my_robot_{}'.format(self.agent_num), 'my_robot_{}'.format(self.agent_num), init_pos_robot)
            self.manager.create_robot('person_{}'.format(self.agent_num), 'person_{}'.format(self.agent_num), init_pos_person)


        self.person = Robot('person_{}'.format(self.agent_num), init_pos= init_pos_person,
                            manager=self.manager, node=self.node,  max_angular_speed=0.7, max_linear_speed=.4)

        self.robot = Robot('my_robot_{}'.format(self.agent_num), init_pos= init_pos_robot,
                            manager=self.manager, node=self.node, max_angular_speed=2.0, max_linear_speed=1.4, relative=self.person, action_mode=self.action_mode_)

    def find_random_point_in_circle(self, radious, min_distance, around_point):
        max_r = 2
        r = (radious - min_distance) * math.sqrt(random.random()) + min_distance
        theta = random.random() * 2 * math.pi
        x = around_point[0] + r * math.cos(theta)
        y = around_point[1] + r * math.sin(theta)
        return (x, y)


    def init_simulator(self):

        self.number_of_steps = 0
        self.node.get_logger().debug("init simulation called")
        self.is_pause = True
        if not self.test_simulation_:

            init_pos_robot, init_pos_person = self.get_init_pos_robot_person()
            self.center_pos_ = init_pos_person["pos"]
            self.current_obsevation_image_.fill(255)
            self.robot.update(init_pos_robot)
            self.person.update(init_pos_person)
            self.prev_action = (0,0)

        self.robot_color = [255,0,0]
        self.person_color = [0,0,255]
        self.goal_color = [0,255,0]
        self.wait_counter_ = 5

        self.path_finished = False
        self.position_thread = threading.Thread(target=self.path_follower, args=(self.current_path_idx, self.robot,))
        self.position_thread.daemon = True

        self.is_reseting = False
        self.position_thread.start()

        self.robot.reset = False
        self.person.reset = False
        # TODO: comment this after start agent
        # self.resume_simulator()
        self.node.get_logger().debug("init simulation finished")

    def states_cb(self, states_msg):

        if self.state_cb_prev_time is None or self.manager.get_time_sec() - self.state_cb_prev_time < 0.05:
            if self.state_cb_prev_time is None:
                self.state_cb_prev_time = self.manager.get_time_sec()
            return

        if self.state_cb_prev_time is not None and self.manager.get_time_sec() - self.state_cb_prev_time > 0.1:
            self.wait_counter_ = 3

        self.state_cb_prev_time = self.manager.get_time_sec()
        if self.wait_counter_ > 0:
            self.wait_counter_ -= 1
            return

        for model_idx in range(len(states_msg.name)):
            if states_msg.name[model_idx] == self.person.name:
                robot = self.person
            elif states_msg.name[model_idx] == self.robot.name:
                robot = self.robot
            else:
                continue

            pos = states_msg.pose[model_idx]
            robot.state_["position"] = (pos.position.x, pos.position.y)
            euler = quat2euler(pos.orientation.x, pos.orientation.y, pos.orientation.z, pos.orientation.w)
            robot.state_["orientation"] = euler[0]
            robot.add_log((pos.position.x, pos.position.y, euler[0]))
        if self.robot_mode == 1:
            self.robot.update_goal(self.person.calculate_ahead(1.5))
        #else:
        #    self.take_supervised_action()
        for robot in (self.robot, self.person):
            if robot.relative is not None and not robot.relative.reset:
                orientation_rel, position_rel = GazeboEnv.get_relative_heading_position(robot, robot.relative, robot.manager.node)
                if orientation_rel is None or position_rel is None:
                    robot.manager.node.get_logger().debug('orientation or pos is none')
                else:
                    self.node.get_logger().debug("agent: {} pos_rel: {:2.2f} {:2.2f}, orientation: {:2.2f}".format(self.agent_num, position_rel[0], position_rel[1], np.rad2deg(orientation_rel)))
                    robot.relative_orientation_history.add_element(orientation_rel, robot.manager.get_time_sec())
                    robot.relative_pos_history.add_element(position_rel, robot.manager.get_time_sec())

            # get velocity
            twist = states_msg.twist[model_idx]
            linear_vel = twist.linear.x
            angular_Vel = twist.angular.z
            robot.velocity_history.add_element(np.asanyarray((linear_vel, angular_Vel)), robot.manager.get_time_sec())

    def add_observation_to_image(self, pos, color, radious):
        to_image_fun = lambda x: (int((x[0] - self.center_pos_[0])*50+1000), int((x[1] - self.center_pos_[1])*50+1000))
        pos_image = to_image_fun(pos)
        if pos_image[0] >2000 or pos_image[0] < 0 or pos_image[1] >2000 or pos_image[1] < 0:
            self.node.get_logger().error("problem with observation: {}".format(pos_image))
            return
        self.current_obsevation_image_ = cv.circle(self.current_obsevation_image_, (pos_image[0], pos_image[1]), radious, color, -1)

    def darken_all_colors(self):
        darken_fun = lambda x: [max(y-1, 10) for y in x]
        self.robot_color = darken_fun(self.robot_color)
        self.person_color = darken_fun(self.person_color)
        self.goal_color = darken_fun(self.goal_color)



    def update_observation_image(self):
        robot_pos = self.robot.get_pos()
        person_pos = self.person.get_pos()
        current_goal = self.robot.goal
        self.add_observation_to_image(robot_pos, self.robot_color, 2)
        self.add_observation_to_image(person_pos, self.person_color, 2)
        self.add_observation_to_image(current_goal, self.goal_color, 2)
        self.darken_all_colors()


    def get_current_observation_image(self):

        image = self.current_obsevation_image_
        image = image/255.

        return image

    def pause(self):
        self.is_pause = True
        self.person.pause()
        self.robot.pause()

    def resume_simulator(self):
        self.node.get_logger().debug("resume simulator")
        self.is_pause = False
        self.person.resume()
        self.robot.resume()
        self.node.get_logger().debug("resumed simulator")

    def calculate_angle_using_path(self, idx):
        return math.atan2(self.path["points"][idx+1][1] - self.path["points"][idx][1], self.path["points"][idx+1][0] - self.path["points"][idx][0])

    @staticmethod
    def get_global_position(action, center, node):
        while not center.is_current_state_ready():
            if center.reset:
                node.get_logger().warn("reseting so return none in rel pos rel: {} center".format(relative.is_current_state_ready(), center.is_current_state_ready()))
                return (None, None)
            time.sleep(0.01)
            node.get_logger().info ("waiting for observation to be ready")
        #relative_orientation = relative.state_['orientation']
        center_pos = np.asarray(center.state_['position'])
        center_orientation = center.state_['orientation']

        pos = [x * 5 for x in action]

        relative_pos = np.asarray(pos)
        # transform the relative to center coordinat
        rotation_matrix = np.asarray([[np.cos(center_orientation), np.sin(center_orientation)], [-np.sin(center_orientation), np.cos(center_orientation)]])
        relative_pos = np.matmul(relative_pos, rotation_matrix)
        global_pos = np.asarray(relative_pos + center_pos)
        return global_pos

    @staticmethod
    def get_relative_position(pos, center, node):
        while not center.is_current_state_ready():
            if center.reset:
                node.get_logger().warn("reseting so return none in rel pos rel: {} center".format(relative.is_current_state_ready(), center.is_current_state_ready()))
                return (None, None)
            time.sleep(0.01)
            node.get_logger().info ("waiting for observation to be ready")
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
    def get_relative_heading_position(relative, center, node):
        while not relative.is_current_state_ready() or not center.is_current_state_ready():
            if relative.reset:
                node.get_logger().debug("reseting so return none in rel pos rel: {} center".format(relative.is_current_state_ready(), center.is_current_state_ready()))
                return (None, None)
            time.sleep(0.01)
            node.get_logger().debug ("waiting for observation to be ready")
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

    def set_robot_to_auto(self):
        self.robot_mode = 2

    """
    the function will check the self.robot_mode:
        0: will not move robot
        1: robot will try to go to a point after person
    """
    def path_follower(self, idx_start, robot):

        counter = 0
        while self.is_pause:
            if self.is_reseting:
                self.node.get_logger().debug( "path follower return as reseting ")
                return
            time.sleep(0.001)
            if counter > 10000:
                self.node.get_logger().debug( "path follower waiting for pause to be false")
                counter = 0
            counter += 1
        self.node.get_logger().debug( "path follower waiting for lock pause:{} reset:{}".format(self.is_pause, self.is_reseting))
        if self.lock.acquire(timeout=10):
            self.node.get_logger().debug("path follower got the lock")
            if self.test_simulation_:
                mode_person = -1
            elif self.is_evaluation_:
                mode_person = 2
            else:
                mode_person = 4#random.randint(0, self.max_mod_person_)
            # if mode_person == 0:
            #     person_thread = threading.Thread(target=self.person.go_to_goal, args=())
            #     person_thread.start()
            if self.robot_mode >= 1:
                self.robot_thread = threading.Thread(target=self.robot.go_to_goal, args=())
                self.robot_thread.start()

            for idx in range (idx_start, len(self.path["points"])-3):
                point = (self.path["points"][idx][0], self.path["points"][idx][1])
                self.current_path_idx = idx
                counter_pause = 0
                while self.is_pause:
                    counter_pause+=1
                    self.node.get_logger().debug("pause in path follower")
                    if self.is_reseting or counter_pause > 200:
                        # if mode_person == 0:
                        #     person_thread.join()
                        self.lock.release()
                        return
                    time.sleep(0.001)
                try:
                    if mode_person != 0:
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
                    self.node.get_logger().error("path follower {}, {}".format(self.is_reseting, e))
                    break
                if self.is_reseting:
                    self.person.stop_robot()
                    break
            self.lock.release()
            self.node.get_logger().debug("path follower release the lock")
            self.path_finished = True
        else:
            self.node.get_logger().debug("problem in getting the log in path follower")
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
                self.node.get_logger().debug("wait for laser scan to get filled sec: {}/25".format(counter / 10))
        if counter>=250:
            raise RuntimeError(
                'exception while calling get_laser_scan:')


        images = np.asarray(images)

        return (images.reshape((images.shape[1], images.shape[2], images.shape[0])))

    def visualize_observation(self, poses, headings, laser_scans):
        image = np.zeros((500, 500, 3))
        for idx in range(len(poses)):
            pt1 = (poses[idx][0] * 50 + 250, poses[idx][1] * 50 + 250)
            pt2 = (pt1[0] + math.cos(headings[idx]) * 30, pt1[1] + math.sin(headings[idx]) * 30)

            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))

            pos_goal, angle_distance = self.get_goal_person()
            pt_goal = np.asarray(angle_distance)
            pt_goal = (int(math.sin(pt_goal[0])* pt_goal[1] * 50 + 250), int(math.cos(pt_goal[0]) * pt_goal[1] * 50 + 250))
            cv.circle(image, pt1, 5, (255, 255 - idx*40, idx*40))

            # cv.arrowedLine(image, pt1, pt2, (0, 0, 255), 5, tipLength=0.6)
            cv.circle(image, pt_goal, 20, (255, 255, 0))
            # cv.imshow("d", pickle_file[0][0])
            cv.imshow("d", image)
        cv.waitKey(1)

    def get_observation(self):
        # got_laser = False
        # while not got_laser:
        #     try:
        #         laser_all = self.get_laser_scan_all()
        #         got_laser = True
        #     except Exception as e:
        #         self.node.get_logger().error("laser_error reseting")
        #         # self.reset(reset_gazebo = True)
        while self.robot.relative_pos_history.avg_frame_rate is None or self.robot.velocity_history.avg_frame_rate is None or self.person.velocity_history.avg_frame_rate is None:
            if self.is_reseting:
                return None
            time.sleep(0.001)
            self.node.get_logger().debug("waiting to get pos/vel pos: {} vel: {} vel_person: {}".format(self.robot.relative_pos_history.avg_frame_rate ,self.robot.velocity_history.avg_frame_rate, self.person.velocity_history.avg_frame_rate))
        pose_history = np.asarray(self.robot.relative_pos_history.get_elemets()).flatten()/5.0
        heading_history = np.asarray(self.robot.relative_orientation_history.get_elemets())/math.pi
        # self.visualize_observation(poses, headings, self.get_laser_scan())
        orientation_position = np.append(pose_history, heading_history)
        velocities = np.concatenate((self.person.get_velocity(), self.robot.get_velocity()))
        #self.node.get_logger().info("velociy min: {} max: {} rate_vel: {} avg: {} clock {} vel {}".format(np.min(velocities), np.max(velocities), self.person.velocity_history.avg_frame_rate, self.person.velocity_history.update_rate, self.manager.get_time_sec(), self.person.get_velocity()))
        return np.append(np.append(orientation_position, velocities), self.prev_action)


    def reset_gazebo(self, no_manager=False):

        self.node.get_logger().debug( "reset gazebo")
        subprocess.call('pkill -9 gzclient', shell=True)
        subprocess.call('tmux kill-session -t gazebo', shell=True)
        subprocess.Popen(['tmux', 'new-session', '-d', '-s', 'gazebo'])
        subprocess.Popen(['tmux', 'send-keys', '-t', 'gazebo', 'source ~/.bashrc', 'C-m'])
        subprocess.Popen(['tmux', 'send-keys', '-t', "gazebo", "ros2 launch /home/payam/ros2_ws/src/follow_ahead_rl/worlds/gazebo.launch.py", "C-m"])
        time.sleep(4)
        if not no_manager:
            self.manager.cleanup()
            self.manager = Manager()

    def __del__(self):
        # todo
        return

    def take_action(self, action):
        if self.test_simulation_:
            return
        self.robot.take_action(action, self.person)
        self.update_observation_image()
        return


    def convert_pos_to_action(self, pos):
        relative = GazeboEnv.get_relative_position(pos, self.person, self.node)
        action = [min(max(x/5., -1), 1) for x in relative]
        return action

    def get_supervised_action(self):
        while not self.person.is_current_state_ready() and not self.is_reseting:
            time.sleep(0.1)
        if self.is_reseting:
            return np.asarray([0,0])


        pos = self.person.calculate_ahead(1.5)
        action = self.convert_pos_to_action(pos)
        return np.asarray(action)


    def take_supervised_action(self):
        action = self.get_supervised_action()
        #self.robot.update_goal(self.person.calculate_ahead(1.5))
        self.take_action(action)

    def step(self, action):
        self.number_of_steps += 1
        self.take_action(action)
        time.sleep(0.08)
        reward = self.get_reward(action)
        ob = self.get_observation()
        episode_over = False
        rel_person = GazeboEnv.get_relative_heading_position(self.robot, self.person, self.manager.node)[1]

        distance = math.hypot(rel_person[0], rel_person[1])
        if self.path_finished:
            if self.agent_num==0:
                self.node.get_logger().info("path finished")
            episode_over = True
        if self.is_collided():
            episode_over = True
            if self.agent_num==0:
                self.node.get_logger().info('agent: {} collision happened episode over'.format(self.agent_num))
            #reward -= 0.5
        elif distance > 4.5:
            episode_over = True
            #reward -= 0.5
            if self.agent_num==0:
                self.node.get_logger().info('agent: {} max distance happened episode over'.format(self.agent_num))
        elif self.number_of_steps > self.max_numb_steps:
            episode_over = True
            if self.agent_num==0:
                self.node.get_logger().info('agent: {} max number of steps episode over'.format(self.agent_num))
        reward = min(max(reward, -1), 1)
        self.node.get_logger().debug("agent: {} action {} reward {}".format((self.agent_num),action, reward))
        self.prev_action = action
        #reward += 1
        return ob, reward, episode_over, {}

    def is_collided(self):
        rel_person = GazeboEnv.get_relative_heading_position(self.robot, self.person, self.manager.node)[1]
        distance = math.hypot(rel_person[0], rel_person[1])
        if distance < 0.5 or self.robot.is_collided:
            return True
        return False

    def get_goal_person(self):
        pos_person = self.person.get_pos()
        pos_goal = None
        angle_distance = None
        for idx in range(self.current_path_idx + 1, len(self.path["points"]) - 3):
            if math.hypot(pos_person[0]-self.path["points"][idx][0], pos_person[1]-self.path["points"][idx][1]) > 3:
                pos_goal = (self.path[idx][0], self.path[idx][1])
                angle_distance= self.robot.angle_distance_to_point(pos_goal)
                break
        return pos_goal, angle_distance

    def get_reward(self, action):
        reward = 0
        if self.use_goal:
            point_goal = (self.path["points"][self.current_path_idx + 1][0], self.path["points"][self.current_path_idx + 1][1])
            try:
                pos_robot = self.robot.get_pos()
            except Exception as e:
                self.node.get_logger().debug("get pos error in get_reward return 0")
                return 0
            distance_goal = math.hypot(pos_robot[1] - point_goal[1], pos_robot[0] - point_goal[0])
            # reward = max (-distance_goal/3.0, -0.9) + 0.4 # between -0.5,0.4
            robot_pos_heading = np.asarray(
                    [pos_robot[0] + math.cos(self.robot.orientation), pos_robot[1] + math.sin(self.robot.orientation)])
            p23 = math.hypot(robot_pos_heading[1]-point_goal[1], robot_pos_heading[0]-point_goal[0])
            p12 = math.hypot(pos_robot[1]-robot_pos_heading[1], pos_robot[0]-robot_pos_heading[0])
            p13 = math.hypot(point_goal[1]-pos_robot[1], point_goal[0]-pos_robot[0])
            if p12 == 0:
                p12 = 0.000001
            if p13 == 0:
                p13 = 0.000001
            angle_robot_goal = np.rad2deg(math.acos(max(min((p12*p12+ p13*p13- p23*p23) / (2.0 * p12 * p13), 1), -1)))
            reward += ((60 - angle_robot_goal)/120.0)/2 +0.25 # between -0.25,0.5
            angle_robot_person, pos_rel = GazeboEnv.get_relative_heading_position(self.robot, self.person, self.manager.node)
            distance = math.hypot(pos_rel[0], pos_rel[1])

            if distance < 1.5:
                reward -= (1.5-distance)/2.0
            elif distance > 2.5:
                reward -= (distance-2.5)/2.0
            else: # distance between 1.5 to 2.5
                reward += 0.5 - abs(distance-2)/2.0 # between 0.25-0.5
        else:
            angle_robot_person, pos_rel = GazeboEnv.get_relative_heading_position(self.robot, self.person, self.manager.node)
            angle_robot_person = math.atan2(pos_rel[1], pos_rel[0])
            angle_robot_person = np.rad2deg(angle_robot_person)
            distance = math.hypot(pos_rel[0], pos_rel[1])
            # Negative reward for being behind the person
            if self.is_collided():
                reward -= 1
            if distance < 0.3:
                reward -= -1
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

            #control_reward = 0
            # control reward penalize if to much chenge for orientation
            # if abs(self.prev_action[1] - action[1])>0.5:
            #     control_reward = (abs(self.prev_action[1] - action[1])/2 - 0.25)

            # self.node.get_logger().debug("agent: {} control_reward is: {} prv/now: {} {}".format(self.agent_num, control_reward, self.prev_action[1], action[1]))
            #reward -= control_reward/5

            reward = min(max(reward, -1), 1)
            # ToDO check for obstacle
        return reward

    def clock_cb(self, msg):
        if self.manager is not None:
            self.manager.current_time = msg.clock.sec + msg.clock.nanosec*0.000000001

    def observation_after_reset_done(self):
        while self.is_reseting:
            time.sleep(0.01)
            rospy.get_logger().debug("reset called when was already resetting")

        return self.get_observation()

    def calculate_using_dynamics(self, v_start, v_end, theta_start, theta_end, pos_start, pos_end):
        x_dot_start = v_start * math.cos(theta_start) 
        x_dot_end = v_end * math.cos(theta_end) 
        y_dot_start = v_start * math.sin(theta_start) 
        y_dot_end = v_end * math.sin(theta_end) 
        start = time.time()
        T = 1
        M = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0],[1, T, T**2, T**3], [0, 1, 2*T, 3*T**2]])
        
        # RHS of constraints in flat output space
        x_constr = np.asarray([[pos_start[0]], [x_dot_start], [pos_end[0]], [x_dot_end]])
        y_constr = np.asarray([[pos_start[1]], [y_dot_start], [pos_end[1]], [y_dot_end]])
        
        # Solve for coefficients of basis functions
        b0 = np.linalg.solve(M, x_constr)
        b1 = np.linalg.solve(M, y_constr)
        # Note: lengths of b0 and b1: 4
        # Computing state and control trajectories from flat outputs
        dt = 0.01;
        t = np.arange(0, T+0.01, dt);
        print (t)
        # First, we compute x, y, and their derivatives; these are analytic since
        # the basis functions are monomials
        x = np.zeros(np.size(t))
        y = np.zeros(np.size(t))
        # x and y have powers of t from 0 to length(b0)-1
        for i in range (np.size(b0)):
            x = x + b0[i] * np.power(t, i)
            y = y + b1[i] * np.power(t, i)
        xdot = np.zeros(np.size(t))
        ydot = np.zeros(np.size(t))
        # xdot and ydot have powers of t from 0 to length(b0)-2
        for i in range (np.size(b0)-1):
            xdot = xdot + (i+1)*b0[i+1]*np.power(t, i)
            ydot = ydot + (i+1)*b1[i+1]*np.power(t, i)
        
        xddot = np.zeros(np.size(t))
        yddot = np.zeros(np.size(t))
        
        # xddot and yddot have powers of t from 0 to length(b0)-3
        for i in range (np.size(b0)-2):
            xddot = xddot + np.math.factorial(i+2)*b0[i+2]*np.power(t, i)
            yddot = yddot + np.math.factorial(i+2)*b1[i+2]*np.power(t, i)
         
        # Finally, we have the theta and v trajectories (x and y trajectories have
        # already been computed)
        theta = np.arctan2(ydot, xdot)
        v = np.divide(xdot, np.cos(theta))
        
        # Use alternate expression for v when cos(theta) == 0
        # Other option: use v = sqrt(xdot^2 + ydot^2)
        small = 1e-5;
        cos_zero_inds = np.absolute(np.cos(theta)) < small
        v[cos_zero_inds] = np.divide(ydot[cos_zero_inds], np.sin(theta[cos_zero_inds]))
         
        # Control trajectories
        omega = np.divide((np.multiply(yddot, xdot) - np.multiply(xddot, ydot)), np.power(v, 2))
        a = np.divide((np.multiply(xdot, xddot) + np.multiply(ydot, yddot)), v)
        print ("time: {}", time.time()-start)
        
        # ax = plt.subplot(4,1,1) 
        # line, = ax.plot(x, y, 'go-',  markersize=1)
        # 
        # 
        # plt.subplot(4, 1, 2)
        # plt.plot(t, omega, 'o-', markersize=1)
        # plt.title('')
        # plt.ylabel('omega')
        # 
        # plt.subplot(4, 1, 3)
        # plt.plot(t, v, '.-', markersize=1)
        # plt.xlabel('time (s)')
        # plt.ylabel('v')
        # 
        # ax = plt.subplot(4, 1, 4)
        # ax.plot(t, theta, '.-', label="theta", markersize=1)
        # ax.plot(t, x, '.-', label="x", markersize=1)
        # ax.plot(t, y, '.-', label="y", markersize=1)
        # plt.legend()
        # plt.xlabel('time (s)')
        # plt.ylabel('theta')
        # 
        # plt.show()

    def reset(self, reset_gazebo=False):

        if self.is_reseting:
            return self.observation_after_reset_done()

        self.is_pause = True
        self.is_reseting = True
        self.robot.reset = True
        self.person.reset = True
        self.node.get_logger().debug("trying to get the lock for reset")
        # if reset_gazebo:
        #     self.reset_gazebo()
        with self.lock:

            self.node.get_logger().debug("got the lock")
            # self.node.get_logger().info("got the lock")
            # try:
            #     # self.manager.pause()
            #     self.robot.remove()
            #     self.person.remove()
            #     self.node.get_logger().info("robot and person removed ")
            #     if reset_gazebo:
            #         self.reset_gazebo()    # self.manager.unpause()
            # except RuntimeError as r:
            #     self.reset_gazebo()
            #     # self.manager.unpause()
            not_init = True
            try:
                if self.position_thread.isAlive():

                    self.node.get_logger().debug("wait for position thread to join")
                    self.position_thread.join()
                    self.node.get_logger().debug("position thread joined")
                if self.robot_mode >= 1 and self.robot_thread is not None and self.robot_thread.isAlive():
                    self.robot_thread.join()
                if self.is_evaluation_:
                    if self.log_file is not None:
                        pickle.dump({"person_history":self.person.log_history, "robot_history":self.robot.log_history}, self.log_file)
                        self.log_file.close()

                    self.path_idx += 1
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
                self.node.get_logger().error("error happend reseting: {}".format(e))
        if not_init:
            self.node.get_logger().info("not init so run reset again")
            return (self.reset())
        else:
            return self.get_observation()

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        return


def test_env():
    gazebo = GazeboEnv()
    print("init gazebo done")

    gazebo.is_evaluation_ = True
    gazebo.set_agent(0)
    print("done set agent")
    gazebo.set_robot_to_auto()
    print("done reset")
    gazebo.resume_simulator()
    while True:
        time.sleep(0.001)
        gazebo.take_supervised_action()



#test_env()
