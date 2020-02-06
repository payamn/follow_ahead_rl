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
from geometry_msgs.msg import PoseStamped

from gazebo_msgs.srv import SetModelState



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

    def calculate_ahead(self, distance):
        pos = Point()
        pos.x = self.pos.x + math.cos(self.orientation) * distance
        pos.y = self.pos.y + math.sin(self.orientation) * distance
        return pos

    def log_pose(self):
        rospy.loginfo("{} position: {} orientation: {} vel(l,a): {}, {}".format(self.name, (self.pos.x, self.pos.y), self.orientation, self.linear_vel, self.angular_vel))



class BaseLine:
    def __init__(self):
        rospy.init_node('follow_ahead_base_line', anonymous=True)
        self.reset = True
        self.robot = Robot("robot")
        self.person = Robot("person")
        self.cmd_vel_pub = rospy.Publisher('/turtlebot1/cmd_vel', Twist, queue_size=10)
        self.simple_goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.last_update_goal = rospy.Time.now().to_sec()
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_cb)
        x = threading.Thread(target=self.person_tracjetories, args=("person_trajectories",))


    def person_tracjetories(self, file_address):
        pass


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
#                self.person.log_pose()
                now = rospy.Time.now().to_sec()
                if (abs(self.last_update_goal - now) > 0.2):
                    pose_stamped = PoseStamped()
                    pose_stamped.header.stamp = rospy.Time.now()
                    pose_stamped.header.frame_id = "odom"
                    pose_stamped.pose.position = self.person.calculate_ahead(1.5)
                    pose_stamped.pose.orientation = msg.pose[i].orientation
                    self.simple_goal_pub.publish(pose_stamped)
                    self.last_update_goal = rospy.Time.now().to_sec()
                    rospy.loginfo("publishing ")

            elif msg.name[i] == "robot":
                self.robot.update(pos, orientation, linear_vel, angular_vel)
                # self.robot.log_pose()
if __name__ == '__main__':
    #wandb.init(project="followahead_dp")
    bl = BaseLine()
    # wandb.init(project="followahead_rldp")
    parser = argparse.ArgumentParser(description='input weight file of the network')
    rospy.spin()

