#!/usr/bin/env python
import argparse
import copy
import traceback

from os import listdir
from os.path import isfile, join

#from cv_bridge import CvBridge


import math
import matplotlib.pyplot as plt
import pandas as pd

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

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from geometry_msgs.msg import Point32
from geometry_msgs.msg import TransformStamped
from rosgraph_msgs.msg import Clock

from costmap_converter.msg import ObstacleArrayMsg
from costmap_converter.msg import ObstacleMsg
from geometry_msgs.msg import Twist


import threading


import _thread

from squaternion import quat2euler
from squaternion import euler2quat

from simple_pid import PID

import pickle
import utils

import logging

logger = logging.getLogger(__name__)


class Robot():
  def __init__(self, name):
    self.name = name
    self.prev_call_vicon = None
    self.state_ = {"position":(None, None), \
                   "orientation":None}
    self.all_states_ = []
    self.last_time_observation = None
    if self.name == "robot":
        rospy.Subscriber("/vicon/Robot/Robot", TransformStamped, self.vicon_cb)
    elif self.name == "person":
        rospy.Subscriber("/vicon/Person/Person", TransformStamped, self.vicon_cb)

  def get_pos(self, idx):
    if "position" in self.all_states_[idx].keys():
      pos = self.all_states_[idx]["position"]
    else:
      pos = self.all_states_[idx]["pos"]
    return pos
  
  def get_orientation(self, idx):
    return self.all_states_[idx]["orientation"]


  def vicon_cb(self, pose_msg):
    if self.last_time_observation is not None and abs(rospy.Time.now().to_sec() - self.last_time_observation) <0.025:
      return
    pos = pose_msg.transform.translation
    self.last_time_observation = rospy.Time.now().to_sec()
    self.state_["position"] = (pos.x, pos.y)
    euler = quat2euler(pose_msg.transform.rotation.x, pose_msg.transform.rotation.y, pose_msg.transform.rotation.z, pose_msg.transform.rotation.w)
    self.state_["orientation"] = euler[0]
    self.all_states_.append(self.state_.copy())

  def get_relative_position(self, center, idx):
    relative_orientation = self.all_states_[idx]['orientation']
    center_pos = np.asarray(center.get_pos(idx))
    center_orientation = center.all_states_[idx]['orientation']

    # transform the pos to center coordinat
    relative_pos = np.asarray(self.get_pos(idx) - center_pos)
    rotation_matrix = np.asarray([[np.cos(-center_orientation), np.sin(-center_orientation)], [-np.sin(-center_orientation), np.cos(-center_orientation)]])
    relative_pos = np.matmul(relative_pos, rotation_matrix)

    return relative_pos

  def get_relative_heading_position(self, center, idx):
    relative_orientation = self.all_states_[idx]['orientation']
    center_pos = np.asarray(center.get_pos(idx))
    center_orientation = center.all_states_[idx]['orientation']
    print (np.rad2deg(relative_orientation - center_orientation))

    # transform the relative to center coordinat
    relative_pos = np.asarray(self.get_pos(idx) - center_pos)
    relative_pos2 = np.asarray((relative_pos[0] +math.cos(relative_orientation) , relative_pos[1] + math.sin(relative_orientation)))
    rotation_matrix = np.asarray([[np.cos(-center_orientation), np.sin(-center_orientation)], [-np.sin(-center_orientation), np.cos(-center_orientation)]])
    relative_pos = np.matmul(relative_pos, rotation_matrix)
    relative_pos2 = np.matmul(relative_pos2, rotation_matrix)
    angle_relative = np.arctan2(relative_pos2[1]-relative_pos[1], relative_pos2[0]-relative_pos[0])
    return angle_relative, relative_pos

  def is_bag_finish(self):
    if self.last_time_observation is not None and abs(rospy.Time.now().to_sec() - self.last_time_observation) > 1:
      return True
    return False

class Results():
  def __init__(self):
    self.center_pos_ = (0, 0)
    self.name = ""
    self.DESIRE_DISTANCE = 1.5
    self.colors_visualization = cv.cvtColor(cv.applyColorMap(np.arange(0, 255, dtype=np.uint8), cv.COLORMAP_WINTER), cv.COLOR_RGB2BGR).reshape(255,3).tolist()
    self.current_obsevation_image_ = np.zeros([500,500,3])
    self.current_obsevation_image_.fill(255)

    self.color_index = 0
    self.first_call_observation = True
    self.robot = Robot("robot")
    self.person = Robot("person")

  def add_line_observation_to_image(self, pos, pos2):
      color = self.colors_visualization[self.color_index]
      pos_image = utils.to_image_coordinate(pos, self.center_pos_)
      pos_image2 = utils.to_image_coordinate(pos2, self.center_pos_)
      if pos_image[0] >self.current_obsevation_image_.shape[0] or pos_image[0] < 0 or pos_image[1] >self.current_obsevation_image_.shape[1] or pos_image[1] < 0:
          rospy.logerr("problem with observation: {}".format(pos_image))
          return
      self.new_obsevation_image_ = cv.line(self.new_obsevation_image_, (pos_image[0], pos_image[1]), (pos_image2[0], pos_image2[1]), color, 1)

  def add_triangle_observation_to_image(self, pos, orientation):
      color = self.colors_visualization[self.color_index]
      pos_image = utils.to_image_coordinate(pos, self.center_pos_)
      pos_triangle1 = utils.to_image_coordinate((pos[0]+math.cos(orientation)*0.3, pos[1]+math.sin(orientation)*0.3), self.center_pos_)
      pos_triangle2 = utils.to_image_coordinate((pos[0]+math.cos(orientation+math.pi/2)*0.1, pos[1]+math.sin(orientation+math.pi/2)*0.1), self.center_pos_)
      pos_triangle3 = utils.to_image_coordinate((pos[0]+math.cos(orientation-math.pi/2)*0.1, pos[1]+math.sin(orientation-math.pi/2)*0.1), self.center_pos_)
      poses = [pos_triangle1, pos_triangle2, pos_triangle3]

      for pos in poses:
          if pos[0] >self.current_obsevation_image_.shape[0] or pos[0] < 0 or pos[1] >self.current_obsevation_image_.shape[1] or pos[1] < 0:
              rospy.logerr("problem with observation: {}".format(pos))
              return
      self.new_obsevation_image_ = cv.drawContours(self.new_obsevation_image_, [np.asarray(poses)], 0, color, -1)


  def add_arrow_observation_to_image(self, pos, orientation):
      color = self.colors_visualization[self.color_index]
      pos_image = utils.to_image_coordinate(pos, self.center_pos_)
      pos_image2 = utils.to_image_coordinate((pos[0]+math.cos(orientation)*0.3, pos[1]+math.sin(orientation)*0.3), self.center_pos_)
      if pos_image[0] >self.current_obsevation_image_.shape[0] or pos_image[0] < 0 or pos_image[1] >self.current_obsevation_image_.shape[1] or pos_image[1] < 0:
          rospy.logerr("problem with observation: {}".format(pos_image))
          return
      self.new_obsevation_image_ = cv.arrowedLine(self.new_obsevation_image_, (pos_image[0], pos_image[1]), (pos_image2[0], pos_image2[1]), color, 2, tipLength=0.5)

  def add_circle_observation_to_image(self, pos, center_pos=None, image=None):
      color = self.colors_visualization[self.color_index]
      if image is None:
          image = self.new_obsevation_image_
      if center_pos is None:
          center_pos = self.center_pos_
      pos_image = utils.to_image_coordinate(pos, center_pos)
      if pos_image[0] >self.current_obsevation_image_.shape[0] or pos_image[0] < 0 or pos_image[1] >self.current_obsevation_image_.shape[1] or pos_image[1] < 0:
          rospy.logerr("problem with observation: {}".format(pos_image))
          return
      return (cv.circle(image , (pos_image[0], pos_image[1]), 4, color, 2))



  def update_observation_image(self, idx):
      self.new_obsevation_image_ = np.copy(self.current_obsevation_image_)
      robot_pos = self.robot.get_pos(idx)
      robot_orientation = self.robot.get_orientation(idx)
      person_pos = self.person.get_pos(idx)
      person_orientation = self.person.get_orientation(idx)
      if person_orientation is None or robot_orientation is None:
          rospy.logerr("person or robot orientation is None")
          return
      if self.first_call_observation:
          self.first_call_observation = False
          self.center_pos = person_pos
      #self.add_circle_observation_to_image(robot_pos)
      self.add_arrow_observation_to_image(robot_pos, robot_orientation)
      self.add_triangle_observation_to_image(person_pos, person_orientation)

      # self.add_line_observation_to_image(robot_pos, person_pos)
      alpha = 0.50
      self.current_obsevation_image_ = cv.addWeighted(self.new_obsevation_image_, alpha, self.current_obsevation_image_, 1 - alpha, 0)
      self.color_index += 4


  def get_current_observation_image(self):

      image = self.current_obsevation_image_.astype(np.uint8)
      #image = image/255.

      return image


  def get_angle_person_robot(self, idx):
    pos_rel = self.robot.get_relative_position(self.person, idx)
    angle_robot_person = math.atan2(pos_rel[1], pos_rel[0])
    return (utils.wrap_pi_to_pi(angle_robot_person))

  def get_dist_person_robot(self, idx):
    pos_rel = self.robot.get_relative_position(self.person, idx)
    return math.hypot(pos_rel[0], pos_rel[1])
  
  def get_reward(self, idx):
    reward = 0
    pos_rel = self.robot.get_relative_position(self.person, idx)
    angle_robot_person = math.atan2(pos_rel[1], pos_rel[0])
    angle_robot_person = np.rad2deg(utils.wrap_pi_to_pi(angle_robot_person))
    distance = math.hypot(pos_rel[0], pos_rel[1])
    # Negative reward for being behind the person
    if distance<0.4:
      reward -= 1
    if distance < 0.5:
      reward = -1.3
    elif abs(distance - self.DESIRE_DISTANCE) < 0.5:
      reward += 0.5 * (0.5 - abs(distance - self.DESIRE_DISTANCE))
    elif distance >= self.DESIRE_DISTANCE + 0.5:
      reward -= 0.25 * (distance - self.DESIRE_DISTANCE + 0.5)
    elif distance < self.DESIRE_DISTANCE - 0.5:
      reward -= (self.DESIRE_DISTANCE - 0.5 - distance)/(self.DESIRE_DISTANCE - 0.5)
    if abs(angle_robot_person) < 25:
      reward += 0.5 * (25 - abs(angle_robot_person)) / 25
    else:
      reward -= 0.25 * abs(angle_robot_person) / 180
    if abs(distance - self.DESIRE_DISTANCE) < 0.5 and abs(angle_robot_person) < 25:
      reward += 0.25

    reward = min(max(reward, -1), 1)
    return reward

  def save(self, name):
    dic_data = {"name":name,"robot":self.robot.all_states_, "person":self.person.all_states_}
    with open (name+"_.pkl", "wb") as f:
      pickle.dump(dic_data, f)

  def load(self, file_address, use_sim=False):
    with open(file_address, "rb") as f:
      dic_data = pickle.load(f)
      
    self.name = dic_data["name"]
    self.person.all_states_ = dic_data["person"][-12:].copy()
    self.robot.all_states_ = dic_data["robot"][-12:].copy()
    if use_sim:
      self.person.all_states_ = [ self.person.all_states_[idx*10] for idx in range (len(self.person.all_states_)//10)] 
      self.robot.all_states_ = [ self.robot.all_states_[idx*10] for idx in range (len(self.robot.all_states_)//10)] 

  def wait_until_bag_finish(self):
    while not self.robot.is_bag_finish() or not self.person.is_bag_finish():
      rospy.sleep(0.1)
      rospy.loginfo("waiting for bag to finish")
      if len(self.person.all_states_)>0 and len(self.robot.all_states_)>0:
        print(self.robot.get_relative_position(self.person, -1))
        print(np.rad2deg(self.get_angle_person_robot(-1)))
    print (self.robot.all_states_)
    print (self.person.all_states_)

  def calculate_orientation_dif(self, idx):
    ori_rel, pos_rel = self.robot.get_relative_heading_position(self.person, idx)
    return ori_rel

  def get_metrics(self):
    rewards = []
    orientations = []
    orientation_dif = []
    distances = []
    len_data = min(len(self.robot.all_states_), len(self.person.all_states_))
    for idx in range (len_data):
      # if idx % 10==0:
      #   self.update_observation_image(idx)
      rewards.append(self.get_reward(idx))
      distances.append(self.get_dist_person_robot(idx))
      orientations.append(self.get_angle_person_robot(idx))
      orientation_dif.append(self.calculate_orientation_dif(idx))

    mean_orientation = np.mean(orientations)
    sum_orientations_m = 0
    for orientation in orientations:
      sum_orientations_m += np.power(utils.wrap_pi_to_pi(mean_orientation - orientation),2)
    sum_orientations_m /= len(orientations)
    std = np.sqrt(sum_orientations_m)

      
    return {"name":self.name, "orientation_mean":np.average(orientations), "orientation_std":std, \
            "reward":np.sum(rewards), "distance":np.average(distances), "distance_std":np.std(distances),\
            "ori_dif":np.average(orientation_dif)}


  def plot_calculate_metrics(self):
    rewards = []
    orientations = []
    distances = []
    len_data = min(len(self.robot.all_states_), len(self.person.all_states_))
    for idx in range (len_data):
      if idx % 3==0:
        self.update_observation_image(idx)
      rewards.append(self.get_reward(idx))
      distances.append(self.get_dist_person_robot(idx))
      orientations.append(self.get_angle_person_robot(idx))
    print (np.rad2deg(self.robot.get_relative_heading_position(self.person, 0)[0]))

    img = self.get_current_observation_image()
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    print(f"\n\ndist avg: {np.average(distances)} orientation avg: {np.rad2deg(np.average(orientations))},  reward: {np.sum(rewards)} reward avg: {np.average(rewards)}")
    cv.imshow("image", img)
    cv.waitKey(0)



def plot_all_results( results, is_sim=False):
  
  name = []
  orientations = []
  rewards = []
  distances = []
  orientations_std = []
  distances_std = []
  for result in results:
    met = result.get_metrics()
    name.append(met["name"])
    rewards.append(met["reward"])
    distances.append(met["distance"])
    distances_std.append(met["distance_std"])
    orientations.append(np.rad2deg(met["orientation_mean"]))
    orientations_std.append(np.rad2deg(met["orientation_std"]))
    print (f"{name[-1]}: Distance_avg: {distances[-1]:.2f} Distance_std: {distances_std[-1]:.2f} Orientation_avg: {orientations[-1]:.1f} Orientation_std: {orientations_std[-1]:.1f} reward: {rewards[-1]:.2f} ori_dif: {np.rad2deg(met['ori_dif']):0.2f}")
    if is_sim:
      print (f"{name[-1]}: ${distances[-1]:.2f}\pm{distances_std[-1]:.1f}$ & ${orientations[-1]:.1f}\pm{orientations_std[-1]:.1f}$ & ${rewards[-1]:.2f}$")
    else:
      print (f"{name[-1]}: ${distances[-1]:.2f}\pm{distances_std[-1]:.1f}$ & ${orientations[-1]:.1f}\pm{orientations_std[-1]:.1f}$ & ${rewards[-1]:.2f}$")
    print ("\n")
    
  #df = pd.DataFrame({'name': name, 'assess':[x for x in range(len(name))]})

  #plt.errorbar(range(len(df['name'])), orientations, orientations_std,  fmt='o')
  #plt.xticks(range(len(df['name'])), df['name'])

if __name__== "__main__":
  parser = argparse.ArgumentParser(description='input weight file of the network')
  parser.add_argument('--name', default="no_name", type=str, help='name_traj')
  parser.add_argument('--file-name', default="no_name", type=str, help='name_file_to_load')
  parser.add_argument('--folder-name', default="no_name", type=str, help='name_file_to_load')
  parser.add_argument('--save', action='store_true')
  parser.add_argument('--load-file', action='store_true')
  parser.add_argument('--load-folder', action='store_true')
  parser.add_argument('--plot', action='store_true')
  parser.add_argument('--use-sim-data', action='store_true')
  parser.add_argument('--from-bag', action='store_true')
  args = parser.parse_args()

  node = rospy.init_node('plot_results')
  if args.load_folder:
    onlyfiles = [join(args.folder_name, f) for f in listdir(args.folder_name) if isfile(join(args.folder_name, f))]
    onlyfiles.sort()

    all_results = []
    for pkl_name in onlyfiles:
      result = Results() 
      result.load(pkl_name)
      name_list = result.name.split("_")
      if not args.use_sim_data and name_list[-1] != "planner" and name_list[-1] != "line":
        print ("error ")
        continue
        new_name = f"{name_list[-1]}_{name_list[-2]}_base_line"
        result.name = new_name
        result.save(new_name)

      all_results.append(result)
    plot_all_results(all_results, args.use_sim_data) 
    #plt.show()

      


  else:
    result = Results()
    if args.from_bag or args.load_file:
      if args.from_bag:
        result.wait_until_bag_finish()
      else:
        result.load(args.file_name, args.use_sim_data)
    else:
      print("exiting you need to load or read from bag file")
      exit(0)

    if args.save:
      result.save(args.name)

    if args.plot:
      result.plot_calculate_metrics()
