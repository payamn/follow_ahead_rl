import time
import cPickle as pickle
import random
import os
import cv2 as cv
import logging
import numpy as np
import matplotlib.pyplot as plt
import threading
import math

from simple_pid import PID

from squaternion import quat2euler
from squaternion import euler2quat

import argparse
import rospy

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped

from gazebo_msgs.srv import SetModelState


class Robot:
    def __init__(self, name):
        self.pos = Point()
        self.orientation = 0
        self.linear_vel = 0
        self.angular_vel = 0
        self.name = name
        self.pos_history = []

    def distance(self, pos):
        return math.hypot(self.pos.x-pos.x, self.pos.y-pos.y)

    def update(self, pos, orientation, linear_vel, angular_vel):
        self.pos = pos
        self.orientation = orientation
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel

        if len(self.pos_history) == 0 or self.distance(self.pos_history[-1]) < 0.1:
            self.pos_history.append(self.pos)

    def calculate_ahead(self, distance):
        pos = Point()
        pos.x = self.pos.x + math.cos(self.orientation) * distance
        pos.y = self.pos.y + math.sin(self.orientation) * distance
        return pos

    def log_pose(self):
        rospy.loginfo("{} position: {} orientation: {} vel(l,a): {}, {}".format(self.name, (self.pos.x, self.pos.y), self.orientation, self.linear_vel, self.angular_vel))



class Trajectories:
    def __init__(self):
        rospy.init_node('follow_ahead_base_line', anonymous=True)
        self.max_angular_vel = 0.5
        self.max_linear_vel = 0.25
        self.angular_pid = PID(0.5, 0, 0.03, setpoint=0)
        self.linear_pid = PID(1.5, 0, 0.05, setpoint=0)
        self.reset = True
        self.person = Robot("person")
        self.robot = Robot("robot")
        self.cmd_vel_pub = rospy.Publisher('/turtlebot1/cmd_vel', Twist, queue_size=10)
        self.simple_goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.last_update_goal = rospy.Time.now().to_sec()
        self.set_model_state_ = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_cb)

        mod = raw_input('select mode, test, or rec:')
        if mod == "test":
            person_test_thread = threading.Thread(target=self.person_follow_path, args=("person_trajectories.pkl",))
            person_test_thread.start()
        else:
            person_trajectories_thread = threading.Thread(target=self.person_tracjetories_recorder, args=("person_trajectories.pkl",))
            person_trajectories_thread.start()

    def create_model_state(self, name, pos, orientation_euler):
        model_state = ModelState()
        model_state.model_name = name
        model_state.pose.position = pos
        quat = euler2quat(0, orientation_euler, 0)
        model_state.pose.orientation.x = quat[3]
        model_state.pose.orientation.y = quat[1]
        model_state.pose.orientation.z = quat[2]
        model_state.pose.orientation.w = quat[0]
        return model_state


    def parse_trajectory(self, trajectory):
        return

    def angle_distance_to_point(self, pos):
        current_pos = self.person.pos
        angle = math.atan2(pos.y - current_pos.y, pos.x - current_pos.x)
        distance = math.hypot(pos.x - current_pos.x, pos.y - current_pos.y)
        angle = (angle - self.person.orientation + math.pi) % (math.pi * 2) - math.pi
        return angle, distance


    def person_follow_path(self, file_address):
        with open('person_trajectories.pkl', 'rb') as file:
            trajectories = pickle.load(file)
        for trajectory in trajectories:
            for name in ["robot", "person"]:
                model_state = self.create_model_state(name, trajectory["start_{}".format(name)]['pos'], trajectory["start_{}".format(name)]['orientation'])
                self.set_model_state_(model_state)
            for point in trajectory["points"]:
                if self.person.distance(point) < 0.2:
                    continue
                while self.person.distance(point) > 0.2:
                    angle, distance = self.angle_distance_to_point(point)
                    angular_vel = -min(max(self.angular_pid(angle)*10, -self.max_angular_vel),self.max_angular_vel)
                    linear_vel = min(max(self.linear_pid(-distance), -self.max_linear_vel), self.max_linear_vel)
                    #if abs(angle) > math.pi /2:
                    linear_vel = linear_vel * math.pow((abs(math.pi - angle)/math.pi), 2)
                    #elif abs(angular_vel) > self.max_angular_vel/2 and linear_vel > self.max_linear_vel/2:
                    #    linear_vel = linear_vel/4

                    cmd_vel = Twist()
                    cmd_vel.linear.x = float(linear_vel)
                    cmd_vel.angular.z = float(angular_vel)
                    self.cmd_vel_pub.publish(cmd_vel)
                    print ("pos {} lineavel {} angularvel {} angle {} distance {}".format((self.person.pos.x, self.person.pos.y), linear_vel, angular_vel, angle, distance))
                    rospy.sleep(0.01)

            name_traj = raw_input('start the next trajectory?')
        print("done all")


    def person_tracjetories_recorder(self, file_address):
        name_traj = raw_input('Enter name to start current trajectory:')
        print (name_traj)
        pickle_data = []
        while name_traj!="end":
            start_pos_robot = {"pos":self.robot.pos, "orientation":self.robot.orientation}
            start_pos_person = {"pos":self.person.pos, "orientation":self.person.orientation}
            ended = raw_input('when finished press enter:')
            pickle_data.append({"start_robot":start_pos_robot, "start_person": start_pos_person, "person_mode": "points", "points": self.person.pos_history, "name":name_traj})
            with open( file_address, "wb" ) as file:
                pickle.dump(pickle_data, file)
            name_traj = raw_input('Enter name of new trajectory to start or enter "end" to finish:')

        print("person trajectories finished")



    def model_states_cb(self, msg):
        for i in range (len(msg.name)):
            if msg.name[i] != "person" and msg.name[i] != "robot":
                continue
            pos = msg.pose[i].position
            euler = quat2euler(msg.pose[i].orientation.x, msg.pose[i].orientation.y, msg.pose[i].orientation.z, msg.pose[i].orientation.w)
            orientation = euler[0]
            linear_vel = msg.twist[i].linear.x
            angular_vel = msg.twist[i].angular.z
            if msg.name[i] == "person":
                self.person.update(pos, orientation, linear_vel, angular_vel)

            elif msg.name[i] == "robot":
                self.robot.update(pos, orientation, linear_vel, angular_vel)

if __name__ == '__main__':
    tr = Trajectories()
    rospy.spin()

