from gym.utils import seeding

import os, subprocess, time, signal

import gym

import math
import random
import _thread

import numpy as np
import cv2 as cv

import rclpy

from squaternion import quat2euler
from squaternion import euler2quat

from ament_index_python.packages import get_package_share_directory

from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

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

    def cleanup(self):
        self.node.destroy_client(self.client_delete )
        self.node.destroy_client(self.client_spawn)
        self.node.destroy_client(self.client_pause)
        self.node.destroy_client(self.client_unpause)
        self.node.destroy_node()

    def get_time_sec(self):
        return self.time.now().nanoseconds/1000000000.0

    def pause(self):
        self.node.get_logger().debug("Connecting to `/client_pause` service...")
        if not self.client_pause.service_is_ready():
            self.client_pause.wait_for_service(timeout_sec=8)
            self.node.get_logger().debug("...connected!")

        request = Empty.Request()
        future = self.client_pause.call_async(request)
        num_try=0
        while num_try<2:
            with self.lock_spin:
                num_try += 1
                self.node.get_logger().debug("Sending service request to `/client_pause`")
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=8)
                if future is not None:
                    self.node.get_logger().warn("paused")
                    return
                else:
                    self.node.get_logger().error("not pause try again ")
        raise RuntimeError(
                'exception while calling service: %r' % future.exception())

    def unpause(self):
        self.node.get_logger().debug("Connecting to `/client_unpause` service...")
        if not self.client_unpause.service_is_ready():
            self.client_unpause.wait_for_service(timeout_sec=8)
            self.node.get_logger().debug("...connected!")

        request = Empty.Request()
        future =  self.client_unpause.call_async(request)
        num_try=0
        while num_try<2:
            num_try+=1
            with self.lock_spin:
                self.node.get_logger().debug("Sending service request to `/client_unpause`")
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=9)
                if future is not None:
                    self.node.get_logger().warn("unpaused")
                    return
                else:
                    self.node.get_logger().error("not pause try again ")
        raise RuntimeError(
                'exception while calling service: %r' % future.exception())


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

    def move_robot(self, name, name_space, translation, rotation):
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
                'exception while calling set_entity state: %r' % future.exception())

    def create_robot(self, name, name_space, translation, rotation):
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
    def __init__(self, window_size, update_time, manager):
        self.data = [None for x in range(window_size)]
        self.idx = 0
        self.manager = manager
        self.update_time = update_time
        self.last_update = 0
        self.window_size = window_size
        # to keep track of last element always in self.idx
        self.update_idx = False

    def add_element(self, element):
        if self.update_idx:
            self.idx = (self.idx + 1) % self.window_size
            self.update_idx = True

        if self.data[self.idx] is None:
            for idx in range (self.window_size):
                self.data[idx] = element
        self.data[self.idx] = element
        if self.manager.get_time_sec() - self.last_update > self.update_time:
            self.last_update = self.manager.get_time_sec()
            self.update_idx = True


    def get_elemets(self):
        return_data = []
        for data in (self.data[self.idx:] + self.data[:self.idx]):
            if data is not None:
                return_data.append(data)

        return return_data

class Robot():
    def __init__(self, name, init_pos, angle, manager, node, max_angular_speed=1, max_linear_speed=1):
        # manager.create_robot(name, name, init_pos, angle)
        manager.move_robot(name, name, init_pos, angle)
        #time.sleep(5)
        self.name = name
        self.manager = manager
        self.init_node = False
        self.node = rclpy.create_node(name)
        self.alive = True
        # self.node = node
        self.init_node = True
        self.deleted = False
        self.collision_distance = 0.5
        self.max_angular_vel = max_angular_speed
        self.max_linear_vel = max_linear_speed
        self.max_laser_range = 5.0 # meter
        self.width_laser_image = 100
        self.height_laser_image = 50
        self.pos_history = History(5, 1, manager)
        self.pos = (None, None)
        qosProfileSensors = rclpy.qos.qos_profile_sensor_data
        
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/{}/cmd_vel'.format(name))
        self.laser_sub = self.node.create_subscription(LaserScan, '/{}/scan'.format(name), self.laser_cb, qos_profile=qosProfileSensors)
        self.model_states_sub = self.node.create_subscription(ModelStates, '/model_states', self.states_cb, qos_profile=qosProfileSensors)
        self.angular_pid = PID(0.1, 0, 0.03, setpoint=0)
        self.linear_pid = PID(0.5, 0, 0.05, setpoint=0)
        self.orientation = angle
        self.orientation_history = History(5, 1, manager)
        self.scan_image = None
        self.scan_image_history = History(5, 1, manager)
        self.is_collided = False
        self.is_pause = False
        self.reset = False
        self.velocity_window = 3
        self.velocity_history = np.zeros((self.velocity_window,2))
        self.last_velocity_idx = 0

        self.spin_thread = threading.Thread(target=self.spin, args=())
        self.spin_thread.start()
        

    def update(self, pos, angle):
        self.manager.move_robot(self.name, self.name, pos, angle)
        self.alive = True
        self.pos_history = History(5, 1, self.manager)
        self.pos = (None, None)
        self.angular_pid = PID(0.1, 0, 0.03, setpoint=0)
        self.linear_pid = PID(1, 0, 0.05, setpoint=0)
        self.orientation = angle
        self.orientation_history = History(5, 1, self.manager)
        self.scan_image = None
        self.scan_image_history = History(5, 1, self.manager)
        self.is_collided = False
        self.is_pause = False
        self.reset = False
        self.velocity_history = np.zeros((self.velocity_window, 2))
        self.last_velocity_idx = 0

    def spin(self):
        while self.alive:
            if self.manager.lock_spin.acquire(timeout=1):
                rclpy.spin_once(self.node)
                self.manager.lock_spin.release()
            time.sleep(0.01)

    def remove(self):
        # self.laser_sub.destroy()
        # self.model_states_sub.destroy()
        self.reset = True
        # self.node.destroy_subscription(self.model_states_sub)
        # self.node.destroy_subscription(self.laser_sub)
        # time.sleep(0.1)
        # self.node.destroy_node()
        # self.laser_sub.destroy()
        # self.model_states_sub.destroy()
        # self.spin_thread.join()
        # try:
        #     self.manager.remove_object(self.name)
        # except RuntimeError as e:
        #     self.node.get_logger().warn("runtime error restart later {} ".format(e))
        #     try:
        #         self.node.destroy_node()
        #     except Exception as e1:
        #         print("error during destroying the node we will raise exception for prev error {} current: ".format(e, e1))
        #     raise RuntimeError("destroy node problem ")
        # self.node.destroy_node()
        # self.node.get_logger().warn("node destroy success")

    def states_cb(self, states_msg):
        model_idx = None
        prev_pos = self.pos
        for i in range(len(states_msg.name)):
            if states_msg.name[i] == self.name:
                # self.node.get_logger().warn("statecb")
                model_idx = i
                break
        if model_idx is None:
            print("cannot find {}".format(self.name))
            return
        pos = states_msg.pose[model_idx]
        self.pos = (pos.position.x, pos.position.y, self.manager.get_time_sec())
        euler = quat2euler(pos.orientation.x, pos.orientation.y, pos.orientation.z, pos.orientation.w)

        self.orientation = euler[0]
        self.orientation_history.add_element(self.orientation)
        
        self.pos_history.add_element(self.pos)

        # get velocity
        twist = states_msg.twist[model_idx]
        linear_vel = twist.linear.x
        angular_Vel = twist.angular.z
        self.velocity_history[self.last_velocity_idx] = np.asarray((linear_vel, angular_Vel))

        # # calculate velocity
        # if prev_pos[0] is not None:
        #     self.velocity_history[self.last_velocity_idx] = math.hypot(self.pos[1]-prev_pos[1], self.pos[0]-prev_pos[0]) / (self.pos[2]-prev_pos[2])
        #     self.last_velocity_idx = (self.last_velocity_idx + 1) % self.velocity_window

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
        #TODO: get velocity from robot states
        return np.mean(self.velocity_history, 0)

    def pause(self):
        self.is_pause = True
        self.stop_robot()

    def resume(self):
        self.is_pause = False

    # action is two digit number the first one is linear_vel (out of 9) second one is angular_vel (our of 6) 
    def take_action(self, action):
        if self.is_pause:
            return
        # linear_vel = (action%10 - 4) * self.max_linear_vel / 5.0
        # angular_vel = (action//10 - 3) * self.max_angular_vel / 3.0
        linear_vel = action[0]*self.max_linear_vel
        angular_vel = action[1]*self.max_angular_vel
        # print ("take action linear {} angular {}".format(linear_vel, angular_vel))
        cmd_vel = Twist()
        cmd_vel.linear.x = float(linear_vel)
        cmd_vel.angular.z = float(angular_vel)
        self.cmd_vel_pub.publish(cmd_vel)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def angle_distance_to_point(self, pos):
        current_pos = self.get_pos()
        if current_pos[0] is None:
            return None, None
        angle = math.atan2(pos[1] - current_pos[1], pos[0] - current_pos[0])
        distance = math.hypot(pos[0] - current_pos[0], pos[1] - current_pos[1])
        angle = (angle - self.orientation + math.pi) % (math.pi * 2) - math.pi
        return angle, distance

    def go_to_pos(self, pos, stop_after_getting=False):
        distance = 2
        # diff_angle_prev = (angle - self.orientation + math.pi) % (math.pi * 2) - math.pi
        while not distance < 1.5:
            if self.is_pause:
                self.stop_robot()
                return
            if self.reset:
                return
            diff_angle, distance = self.angle_distance_to_point(pos)
            if distance is None:
                return
            # if abs(diff_angle_prev - diff_angle) > math.pi*3/:
            #     diff_angle = diff_angle_prev
            # else:
            #     diff_angle_prev = diff_angle
            # angular_vel = min(max(self.angular_pid(math.atan2(math.sin(angle-self.orientation), math.cos(angle-self.orientation)))*200, -self.max_speed/3),self.max_speed)
            angular_vel = -min(max(self.angular_pid(diff_angle)*3, -self.max_angular_vel),self.max_angular_vel)
            linear_vel = min(max(self.linear_pid(-distance), -self.max_linear_vel), self.max_linear_vel)
            if abs(diff_angle) > math.pi /2:
                linear_vel = linear_vel /20.0
            elif abs(angular_vel) > self.max_angular_vel/2 and linear_vel > self.max_linear_vel/2:
                linear_vel = linear_vel/4
            if self.reset:
                return
            cmd_vel = Twist()
            cmd_vel.linear.x = float(linear_vel)
            cmd_vel.angular.z = float(angular_vel)
            self.cmd_vel_pub.publish(cmd_vel) 
            time.sleep(0.005)

        if stop_after_getting:
            self.stop_robot()

    def get_pos(self):
        counter_problem = 0
        while self.pos[0] is None:
            if self.reset:
                return (None, None)
            if counter_problem > 20:
                self.node.get_logger().warn("waiting for pos to be available {}/{}".format(counter_problem/10, 20))
            time.sleep(0.01)
            counter_problem += 1
            if counter_problem > 200:
                raise Exception('Probable shared memory issue happend')

        return self.pos[:2]

  #   def imu_cb(self, imo_msg):
  #       self.orientation = quat2euler(
  #           imo_msg.orientation.x, imo_msg.orientation.y, imo_msg.orientation.z, imo_msg.orientation.w)[0]
  #       self.orientation_history.add_element(self.orientation)

    def get_laser_image(self):
        while self.scan_image is None:
            time.sleep(0.1)
        return np.expand_dims(self.scan_image, axis=2)


class GazeboEnv(gym.Env):

    def __init__(self):

        self.use_goal = False

        self.is_reseting = True
        self.lock = _thread.allocate_lock()
        self.robot_mode = 0

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(17,))
        # gym.spaces.Tuple(
        #     (
        #         gym.spaces.Box(low=0, high=1, shape=(50, 100, 5)),
        #         gym.spaces.Box(low=0, high=1, shape=(17,))
        #     )
        # )
        # Action space omits the Tackle/Catch actions, which are useful
        # on defense
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.min_distance = 1
        self.max_distance = 2.5
        self.max_numb_steps = 2000
        self.reward_range = [0, 2]
    def set_agent(self, agent_num):
        rclpy.init()
        self.node = rclpy.create_node("gazebo_env_{}".format(agent_num))
        # self.reset_gazebo(no_manager=True)
        self.manager = Manager(agent_num)

        # use goal from current path to calculate reward if false use a the heading and relative position  to calculate reward
        # read the path for robot
        self.path = []
        try:
            with open('data/first', 'rb') as f:
                self.path = pickle.load(f)
        except Exception as e:
            print(e)
        self.agent_num = agent_num
        self.create_robots(create_robot_manager=False)
        with self.lock:
            self.init_simulator()

    def spin(self):
        while rclpy.ok():
            while self.is_reseting:
                time.sleep(0.1)
            rclpy.spin_once(self.node)
            time.sleep(0.01)

    def create_robots(self, create_robot_manager=False):
        idx_start = random.randint(0, len(self.path) - 20)
        self.current_path_idx = idx_start

        init_pos_person = self.path[idx_start]
        angle_person = self.calculate_angle_using_path(idx_start)
        idx_robot = idx_start + 1
        while (math.hypot(self.path[idx_robot][1] - self.path[idx_start][1],
                          self.path[idx_robot][0] - self.path[idx_start][0]) < 1.6):
            idx_robot += 1
        angle_robot = self.calculate_angle_using_path(idx_robot)
        # manager.create_robot(name, name, init_pos, angle)
        if create_robot_manager:
            self.manager.create_robot('my_robot_{}'.format(self.agent_num), 'my_robot_{}'.format(self.agent_num), self.path[idx_robot], angle_robot)
            self.manager.create_robot('person_{}'.format(self.agent_num), 'person_{}'.format(self.agent_num), self.path[idx_start], angle_person)

        self.robot = Robot('my_robot_{}'.format(self.agent_num), init_pos=self.path[idx_robot],  angle=angle_robot,
                            manager=self.manager, node=self.node, max_angular_speed=1, max_linear_speed=1)
        self.person = Robot('person_{}'.format(self.agent_num), init_pos=self.path[idx_start],  angle=angle_person,
                            manager=self.manager, node=self.node,  max_angular_speed=0.8, max_linear_speed=.8)
        
    def init_simulator(self):

        self.number_of_steps = 0
        self.node.get_logger().info("init simulation called")
        self.is_reseting = False
        self.is_pause = True
        idx_start = random.randint(0, len(self.path)-20)
        self.current_path_idx = idx_start

        init_pos_person = self.path[idx_start]
        angle_person = self.calculate_angle_using_path(idx_start)
        idx_robot = idx_start + 1
        while (math.hypot (self.path[idx_robot][1] - self.path[idx_start][1], self.path[idx_robot][0] - self.path[idx_start][0]) < 1.6):
            idx_robot += 1
        angle_robot = self.calculate_angle_using_path(idx_robot)
        # self.manager.pause()
        # self.robot = Robot('my_robot', init_pos=self.path[idx_robot],  angle=angle_robot, manager=self.manager, node=self.node, max_angular_speed=1.6, max_linear_speed=1.6)
        # self.person = Robot('person', init_pos=self.path[idx_start],  angle=angle_person, manager=self.manager, node=self.node,  max_angular_speed=1.4, max_linear_speed=.9)
        self.robot.update(self.path[idx_robot],  angle_robot)
        self.person.update(self.path[idx_start],  angle_person)
        # self.manager.unpause()
        self.position_thread = threading.Thread(target=self.path_follower, args=(self.person, idx_start, self.robot,))
        self.position_thread.daemon = True

        self.node.get_logger().info("starting position thread")
        self.position_thread.start()
        # while True:
        #     print(self.get_angle_person_robot())
        #     time.sleep(0.5)


        self.robot.reset = False
        self.person.reset = False
        
        # TODO: comment this after start agent
        # self.resume_simulator()
        self.node.get_logger().info("init simulation finished")

    def pause(self):
        self.is_pause = True
        self.person.pause()
        self.robot.pause()

    def resume_simulator(self):
        self.node.get_logger().info("resume simulator")
        self.is_pause = False
        self.person.resume()
        self.robot.resume()
        self.node.get_logger().info("resumed simulator")

    def calculate_angle_using_path(self, idx):
        return math.atan2(self.path[idx+1][1] - self.path[idx][1], self.path[idx+1][0] - self.path[idx][0])

    def get_relative_heading_position(self, relative, center):
        got_position = False
        while not got_position:
            try:
                person_pos_history = self.person.pos_history.get_elemets()
                person_orientation_history = self.person.orientation_history.get_elemets()
                person_pos = self.person.get_pos()
                robot_pos = self.robot.get_pos()
                person_pos_history = self.person.pos_history.get_elemets()
                person_orientation_history = self.person.orientation_history.get_elemets()
                if (len(person_pos_history) == self.person.pos_history.window_size and len(
                        person_orientation_history) == self.person.orientation_history.window_size):
                    got_position = True
                else:
                    self.node.get_logger().warn(
                        "waiting for person_pos_history and orientation to be filled: {} {}".format(
                            len(person_pos_history), len(person_orientation_history)))
            except Exception as e:
                self.node.get_logger().error(
                    " because of exception (in get_person_position heading rel..), {}".format(e))

        relative_orientation = relative.orientation
        center_pos = np.asarray(center.pos[:2])
        center_orientation = center.orientation

        # transform the relative to center coordinate
        relative_pos = np.asarray(relative.pos[:2] - center_pos)
        relative_pos2 = np.asarray((relative_pos[0] +math.cos(relative_orientation) , relative_pos[1] + math.sin(relative_orientation)))
        rotation_matrix = np.asarray([[np.cos(-center_orientation), np.sin(-center_orientation)], [-np.sin(-center_orientation), np.cos(-center_orientation)]])
        relative_pos = np.matmul(relative_pos, rotation_matrix)
        relative_pos2 = np.matmul(relative_pos2, rotation_matrix)
        angle_relative = np.arctan2(relative_pos2[1]-relative_pos[1], relative_pos2[0]-relative_pos[0])
        return angle_relative, relative_pos

    def get_person_position_heading_relative_robot(self, get_history=False):
        got_position = False
        person_pos_history = self.person.pos_history.get_elemets()
        person_orientation_history = self.person.orientation_history.get_elemets()

        while not got_position:
            need_reset = False
            try:
                person_pos_history = self.person.pos_history.get_elemets()
                person_orientation_history = self.person.orientation_history.get_elemets()
                person_pos = self.person.get_pos()
                robot_pos = self.robot.get_pos()
                person_pos_history = self.person.pos_history.get_elemets()
                person_orientation_history = self.person.orientation_history.get_elemets()
                if (len(person_pos_history) == self.person.pos_history.window_size and len(person_orientation_history) == self.person.orientation_history.window_size):
                    got_position = True
                else: 
                    self.node.get_logger().warn ("waiting for person_pos_history and orientation to be filled: {} {}".format(len(person_pos_history), len(person_orientation_history)))
            except Exception as e:
                self.node.get_logger().error(" because of exception (in get_person_position heading rel..), {}".format(e))
                need_reset = True
            if need_reset:
                # self.reset_gazebo()
                not_init = True
                while not_init:
                    try:
                        self.init_simulator()
                        not_init = False
                    except RuntimeError as e:
                        self.node.get_logger().error("error happend reseting: {}".format(e))
                        # self.reset_gazebo()



        # person_pos_history = self.person.pos_history.get_elemets()
        # person_orientation_history = self.person.orientation_history.get_elemets()

        # while len(person_pos_history) != self.person.pos_history.window_size or len(person_orientation_history) != self.person.orientation_history.window_size:
        #     self.node.get_logger().warn ("waiting for person_pos_history and orientation to be filled: {} {}".format(len(person_pos_history), len(person_orientation_history)))
        #     robot_pos = self.robot.get_pos()
        #     person_pos_history = self.person.pos_history.get_elemets()
        #     person_orientation_history = self.person.orientation_history.get_elemets()
        #     time.sleep(0.1)

        person_pos_and_headings = [(person_pos_history[idx][:2] ,np.asarray((person_pos_history[idx][0] + math.cos(person_orientation_history[idx]), person_pos_history[idx][1] + math.sin(person_orientation_history[idx])))) for idx in range (len(person_pos_history)) ]
        angle_robot = math.pi / 2 - self.robot.orientation
        rotation_matrix = np.asarray(
            [[math.cos(angle_robot), -math.sin(angle_robot)], [math.sin(angle_robot), math.cos(angle_robot)]])
        poses = []
        heading = []
        for person_pos_and_heading in person_pos_and_headings:
            person_poses_transformed = [np.asarray(pos) - np.asarray(robot_pos) for pos in person_pos_and_heading]
            person_poses_rt = [np.dot(rotation_matrix, matrix) for matrix in person_poses_transformed]
            heading_person = np.asarray(math.atan2(person_poses_rt[1][1] - person_poses_rt[0][1],
                                                   person_poses_rt[1][0] - person_poses_rt[0][0]))
            poses.append(person_poses_rt[0])
            heading.append(heading_person)

        if not get_history:
            return((poses[0], heading[0]))

        return poses, heading

    def set_robot_to_auto(self):
        self.robot_mode = 1

    """
    the function will check the self.robot_mode:
        0: will not move robot
        1: robot will try to go to a point after person
    """
    def path_follower(self, person, idx_start, robot):

        counter = 0
        while self.is_pause:
            if self.is_reseting:
                self.node.get_logger().info( "path follower return as reseting ")
                return 
            time.sleep(0.01)
            if counter > 10000:
                self.node.get_logger().info( "path follower waiting for pause to be false")
                counter = 0
            counter += 1
        self.node.get_logger().info( "path follower waiting for lock pause:{} reset:{}".format(self.is_pause, self.is_reseting))
        if self.lock.acquire(timeout=10):
            self.node.get_logger().info("path follower got the lock")
            for idx in range (idx_start, len(self.path)-3):
                point = self.path[idx]
                self.current_path_idx = idx
                counter_pause = 0
                while self.is_pause:
                    counter_pause+=1
                    self.node.get_logger().info("pause in path follower")
                    if self.is_reseting or counter_pause > 200:
                        self.lock.release()
                        return
                    time.sleep(0.1)
                try:
                    person_thread = threading.Thread(target=self.person.go_to_pos, args=(point, True,))
                    person_thread.start()
                    # person.go_to_pos(point)
                    if self.robot_mode == 1:
                        noisy_point = (self.path[idx+3][0] +min(max(np.random.normal(),-0.5),0.5), self.path[idx+3][1] +min(max(np.random.normal(),-0.5),0.5))
                        robot_thread = threading.Thread(target=self.robot.go_to_pos, args=(noisy_point,True,))
                        robot_thread.start()
                        robot_thread.join()

                    person_thread.join()

                except Exception as e:
                    self.node.get_logger().warn("path follower {}, {}".format(self.is_reseting, e))
                    break
                self.node.get_logger().debug("got to point: {} out of {}".format(idx - idx_start, len(self.path) - idx_start ))
                if self.is_reseting:
                    person.stop_robot()
                    break
            self.lock.release()
            self.node.get_logger().info("path follower release the lock")
        else:
            self.node.get_logger().error("problem in getting the log in path follower")
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
                self.node.get_logger().info("wait for laser scan to get filled sec: {}/25".format(counter / 10))
        if counter>=250:
            raise RuntimeError(
                'exception while calling get_laser_scan:')


        images = np.asarray(images)

        return (images.reshape((images.shape[1], images.shape[2], images.shape[0])))

    def get_angle_person_robot(self):
        orientation_person = self.person.orientation
        angle_robot = self.robot.orientation
        robot_person_vec = -self.get_person_position_heading_relative_robot()[0]
        if orientation_person is None:
            return
        heading_person_vec = np.asarray([math.cos(orientation_person), math.sin(orientation_person)])
        dot = np.dot(robot_person_vec, heading_person_vec)
        angle = math.acos(dot / (np.linalg.norm(heading_person_vec) * np.linalg.norm(robot_person_vec)))
        return angle * 180 / math.pi

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
        got_laser = False
        while not got_laser:
            try:
                laser_all = self.get_laser_scan_all()
                got_laser = True
            except Exception as e:
                self.node.get_logger().error("laser_error reseting")
                # self.reset(reset_gazebo = True)
        heading, pose = self.get_relative_heading_position(self.robot, self.person)
        print("heading : {} poses: {}".format(np.rad2deg(heading), pose))
        # self.visualize_observation(poses, headings, self.get_laser_scan())
        orientation_position = np.append(pose, heading)
        velocities = np.concatenate((self.person.get_velocity(), self.robot.get_velocity()))
        
#        return np.append(orientation_position, velocities)
        return heading, pose, self.person.get_velocity(), self.robot.get_velocity()


    def reset_gazebo(self, no_manager=False):
        
        self.node.get_logger().error( "reset gazebo")
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
        self.robot.take_action(action)
        return

    def step(self, action):
        self.number_of_steps += 1
        self.take_action(action)
        time.sleep(0.05)
        reward = self.get_reward()
        ob = self.get_observation()
        episode_over = False
        rel_person = self.get_relative_heading_position(self.robot, self.person)[1]
        distance = math.hypot(rel_person[0], rel_person[1])
        if self.is_collided():
            episode_over = True
            print('collision happened episode over')
            reward -= 0.5
        elif distance > 5:
            episode_over = True
            print('max distance happened episode over')
        elif self.number_of_steps > self.max_numb_steps:
            episode_over = True
            print('max number of steps episode over')
        reward = min(max(reward, -1), 1)
        self.node.get_logger().debug("reward: {} ".format(reward))
        print ("agent: {} ".format(self.agent_num),action, reward)
        #reward += 1
        return ob, reward, episode_over, {}

    def is_collided(self):
        rel_person = self.get_person_position_heading_relative_robot()[0]
        distance = math.hypot(rel_person[0], rel_person[1])
        if distance < 0.8 or self.robot.is_collided:
            return True
        return False

    def get_goal_person(self):
        pos_person = self.person.get_pos()
        pos_goal = None
        angle_distance = None
        for idx in range(self.current_path_idx + 1, len(self.path) - 3):
            if math.hypot(pos_person[0]-self.path[idx][0], pos_person[1]-self.path[idx][1]) > 3:
                pos_goal = self.path[idx]
                angle_distance= self.robot.angle_distance_to_point(pos_goal)
                break
        return pos_goal, angle_distance

    def get_reward(self):
        reward = 0
        if self.use_goal:
            point_goal = self.path[self.current_path_idx + 1]
            try:
                pos_robot = self.robot.get_pos()
            except Exception as e:
                self.node.get_logger().error("get pos error in get_reward return 0")
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
            pos_rel = self.get_person_position_heading_relative_robot()[0]
            distance = math.hypot(pos_rel[0], pos_rel[1])

            if distance < 1.5:
                reward -= (1.5-distance)/2.0
            elif distance > 2.5:
                reward -= (distance-2.5)/2.0
            else: # distance between 1.5 to 2.5
                reward += 0.5 - abs(distance-2)/2.0 # between 0.25-0.5
        else:
            angle_robot_person, pos_rel = self.get_relative_heading_position(self.robot, self.person)
            angle_robot_person = np.rad2deg(angle_robot_person)
            distance = math.hypot(pos_rel[0], pos_rel[1])
            # Negative reward for being behind the person
            if self.is_collided():
                reward -= 1
            if abs(distance - 1.7) < 0.7:
                reward += 0.1 * (0.7 - abs(distance - 1.7))
            elif distance >= 1.7:
                reward -= 0.01 * distance
            elif distance < 1:
                reward -= 1 - distance
            if abs(angle_robot_person) < 45:
                reward += 0.2 * (45 - abs(angle_robot_person)) / 45
            else:
                reward -= 0.3 * abs(angle_robot_person) / 180
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
        self.node.get_logger().info("trying to get the lock for reset")
        # if reset_gazebo:
        #     self.reset_gazebo()
        with self.lock:

            self.node.get_logger().info("got the lock")
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

                    self.node.get_logger().info("wait for position thread to join")
                    self.position_thread.join()
                    self.node.get_logger().info("position thread joined")
                self.init_simulator()
                not_init = False
            except RuntimeError as e:
                self.node.get_logger().error("error happend reseting: {}".format(e))
        if not_init:
            return (self.reset())
        # else:
        #     return self.get_observation()

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        return

# gazebo = GazeboEnv()

def test_env():
    rclpy.init()
    node = rclpy.create_node("gazebo_env")
    manager = Manager()
    robot = Robot('my_robot', init_pos=(4,4,4),  angle=0, manager=manager)
    while rclpy.ok():
        
        rclpy.spin_once(node)
    exit()

# test_env()
