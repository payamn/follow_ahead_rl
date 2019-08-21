import numpy as np
import cv2 as cv

import rospy
import math

from sensor_msgs.msg import NavSatFix

import subprocess
import pickle

robot_service_str = '/my_robot/'
all_pos = []
def position_cb(pos_msg):
    pos = (-pos_msg.longitude, -pos_msg.latitude)
    if len(all_pos) == 0 or math.hypot(all_pos[-1][0]-pos[0], all_pos[-1][1]-pos[1]) > 0.5:
        all_pos.append(pos)
if __name__=="__main__":
    rospy.init_node("record_path", anonymous=True)
    subprocess.call(["mkdir", "data", "-p"])
    pos_sub = rospy.Subscriber(robot_service_str + '/gps/values', NavSatFix, position_cb)
    val = input("Start saving path when finished please enter with name of the file: ")
    with open('data/'+val, 'wb') as f:
        pickle.dump(all_pos, f)
