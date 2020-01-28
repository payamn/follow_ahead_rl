import time
import random
import os
import cv2 as cv
import logging
import numpy as np
import matplotlib.pyplot as plt
import threading
import math

import argparse
import rospy

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist


class Robot:
    def __init__(self, pos, orientation, linear_vel, angular_vel):
        self.pos = pos
        self.orientation = orientation
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel



class BaseLine:
    def __init__(self):
        rospy.init_node('follow_ahead_base_line', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/turtlebot2/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_cb)


    def model_states_cb(self, msg):
        rospy.loginfo("{}".format(msg.name))
if __name__ == '__main__':
    #wandb.init(project="followahead_dp")
    bl = BaseLine()
    # wandb.init(project="followahead_rldp")
    parser = argparse.ArgumentParser(description='input weight file of the network')
    rospy.spin()

