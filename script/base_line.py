import time
import random
import os
import cv2 as cv
import logging
import numpy as np
import matplotlib.pyplot as plt
import threading
import math

from squaternion import quat2euler
from squaternion import euler2quat

import argparse
import rospy

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point


class Robot:
    def __init__(self, name):
        self.pos = Point()
        self.orientation = 0
        self.linear_vel = 0
        self.angular_vel = 0
        self.name = name

    def update(self, pos, orientation, linear_vel, angular_vel):
        self.pos = pos
        self.orientation = orientation
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel

    def log_pose(self):
        rospy.loginfo("{} position: {} orientation: {} vel(l,a): {}, {}".format(self.name, (self.pos.x, self.pos.y), self.orientation, self.linear_vel, self.angular_vel))



class BaseLine:
    def __init__(self):
        rospy.init_node('follow_ahead_base_line', anonymous=True)
        self.robot = Robot("robot")
        self.person = Robot("person")
        self.cmd_vel_pub = rospy.Publisher('/turtlebot2/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_cb)


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
                self.person.log_pose()
            elif msg.name[i] == "robot":
                self.robot.update(pos, orientation, linear_vel, angular_vel)
                self.robot.log_pose()
if __name__ == '__main__':
    #wandb.init(project="followahead_dp")
    bl = BaseLine()
    # wandb.init(project="followahead_rldp")
    parser = argparse.ArgumentParser(description='input weight file of the network')
    rospy.spin()

