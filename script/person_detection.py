# From Python
# It requires OpenCV installed for Python
import sys
import copy
import cv2
import os
from sys import platform
import argparse
import time

from squaternion import quat2euler
from squaternion import euler2quat

import math
import threading
import numpy as np

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud

from cylinder_fitting import fit as cylinder_fit

from message_filters import ApproximateTimeSynchronizer, Subscriber
import rospy
from cv_bridge import CvBridge, CvBridgeError


# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
#parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "/media/payam/data/catkin_ws/src/follow_ahead_rl/script/openpose/models"
params["face"] = False
params["hand"] = False

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item



class PersonDetection:
    def __init__(self, params_op):
        self.bridge = CvBridge()
        self.camera_intrinsic_ = None
        self.lock = threading.Lock()

        self.rgb_subb = rospy.Subscriber("/zed2/zed_node/rgb/image_rect_color", Image, self.rgb_cb)
        #self.rgb_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.camera_cb)
        self.depth_sub = rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, self.depth_cb)
        #ats = ApproximateTimeSynchronizer([self.rgb_subb, self.depth_sub], queue_size=2, slop=2)
        #ats.registerCallback(self.camera_cb)
        self.rgb_msg = None
        self.depth_msg = None

        self.camera_info_sub = rospy.Subscriber("/zed2/zed_node/depth/camera_info", CameraInfo, self.camera_info_cb)

        self.image_debug_pub_ = rospy.Publisher("person_detection", Image)
        self.toros_pub_ = rospy.Publisher('torso', PointCloud, queue_size=10)
        self.marker_pub_ = rospy.Publisher('person_pose', Marker, queue_size=1)

        # starting openpose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params_op)
        self.opWrapper.start()
        self.detection_tread_ = threading.Thread(target=self.camera_thread, args=())
        self.detection_tread_.start()

    def publish_pose_person(self, pos, orientation):
        marker = Marker()
        marker.header.frame_id = "zed2_left_camera_optical_frame"
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.scale.x = 0.3
        marker.scale.y = 0.05
        marker.scale.z = 0.03
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        quaternion_rotation = euler2quat(orientation, 0, 0)
        marker.pose.orientation.x = quaternion_rotation[3]
        marker.pose.orientation.y = quaternion_rotation[1]
        marker.pose.orientation.z = quaternion_rotation[2]
        marker.pose.orientation.w = quaternion_rotation[0]
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        self.marker_pub_.publish(marker)


    def camera_info_cb(self, info_msg):
        if self.camera_intrinsic_ is None:
            camera_intrinsic = info_msg.K
            self.camera_intrinsic_ = {
                    "fx": camera_intrinsic[0],
                    "cx": camera_intrinsic[2],
                    "fy": camera_intrinsic[4],
                    "cy": camera_intrinsic[5]
                    }

            self.camera_info_sub.unregister()

        rospy.loginfo("got camera intrisic {}".format(self.camera_intrinsic_))

    def rgb_cb(self, rgb_msg):
        with self.lock:
            self.rgb_msg = rgb_msg

    def depth_cb(self, depth_msg):
        with self.lock:
            self.depth_msg = depth_msg

    def camera_thread(self):
        while not rospy.is_shutdown():
            while self.rgb_msg is None or self.depth_msg is None:
                rospy.sleep(0.1)
                rospy.logwarn("not available")
                continue
            with self.lock:
                rgb_msg = copy.deepcopy(self.rgb_msg)
                depth_msg = copy.deepcopy(self.depth_msg)
            #if abs(rgb_msg.header.stamp.to_sec() - rospy.Time.now().to_sec()) > 0.1:
            #    rospy.loginfo("skipping {}".format(rgb_msg.header.stamp.to_sec() - rospy.Time.now().to_sec()))
            #    continue
            image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            datum = self.detect(image, depth)
            if datum.poseKeypoints.shape == ():
                rospy.logwarn("open pose not detecting the person")
                rospy.sleep(0.05)
                continue
            person = datum.poseKeypoints[0]
            try:
                #self.torso_detection(person, depth, image)
                self.detect_lines(image, depth, [(person[2][0], person[2][1]), (person[5][0], person[5][1]), (person[9][0], person[9][1]), (person[12][0], person[12][1])])
                self.publish_image(image)
                #self.visualize_points(person, depth, image)
            except Exception as e:
                rospy.logwarn("error: {}".format(e))
            rospy.sleep(0.05)
        #cv2.imshow("img", image)
        #cv2.waitKey(1)

    def publish_image(self, image):
        self.image_debug_pub_.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

    def visualize_points(self, person, depth, image):
        pos_person = {x:None for x in range(16)}
        for i in range(len(person)):
            pos = self.calculate_depth(image, depth, (person[i][0], person[i][1]))
            pos_person[i] = (pos,  person[i])
            cv2.putText(image, '{:1.0f}'.format(i),(int(person[i][0]-10), int(person[i][1]+1)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0) , thickness=4, lineType=cv2.LINE_AA)

    def torso_detection(self, person, depth, image):

        start = time.time()
        torso_img = np.zeros((depth.shape[0],depth.shape[1]), np.uint8)
        cv2.fillPoly(image, [np.int32(np.asarray([(person[2][0], person[2][1]), (person[5][0], person[5][1]) , (person[12][0], person[12][1]),(person[9][0], person[9][1])]))], (255,255, 255))
        cv2.fillPoly(torso_img, [np.int32(np.asarray([(person[2][0], person[2][1]), (person[5][0], person[5][1]) , (person[12][0], person[12][1]),(person[9][0], person[9][1])]))], (255,255, 255))
        masked = np.copy(depth)
        masked =  cv2.bitwise_and(depth, masked, mask=torso_img)
        masked[masked==float('inf')] = 0
        masked[np.isnan(masked)] = 0
        valid_index = np.nonzero(masked)
        median = np.median(masked[valid_index])
        mask_new = np.logical_or(masked<median*0.9,masked>median*1.1)
        masked[mask_new]= 0
        valid_index = np.nonzero(masked)
        point_cloud = PointCloud()
        point_cloud.header.stamp = rospy.Time.now()
        point_cloud.header.frame_id = "zed2_left_camera_optical_frame"
        data_points = []
        for i in range(valid_index[0].shape[0]):
            point = Point32()
            z = masked[valid_index[0][i], valid_index[1][i]]
            point.x = (valid_index[1][i] - self.camera_intrinsic_["cx"]) * z  / self.camera_intrinsic_["fx"]
            point.y = (valid_index[0][i] - self.camera_intrinsic_["cy"]) * z / self.camera_intrinsic_["fy"]
            point.z = z
            #print (z, valid_index[0][i], valid_index[1][i])
            point_cloud.points.append(point)
            data_points.append(np.asarray((point.y, point.x, point.z)))
        #print ("before fit cylinder")
        #x = cylinder_fit(data_points)
        #print ("fit cylinder: {}".format(x))
        self.toros_pub_.publish(point_cloud)
        end = time.time()
        print("torso detection finished. Total time: " + str(end - start) + " seconds")


    def calculate_depth(self, image, depth, point, visualize=True):
        circle_img = np.zeros((depth.shape[0],depth.shape[1]), np.uint8)
        if visualize:
            cv2.circle(image, point, 10, [0,255,0 ], 7)
        cv2.circle(circle_img,(point[0], point[1]),10,255,-1)
        masked = np.copy(depth)
        masked =  cv2.bitwise_and(depth, masked, mask=circle_img)
        masked = masked[masked!=float('inf')]
        masked = masked[masked!=float('nan')]
        masked = masked[masked>0]

        z = np.median(masked)
        #rospy.loginfo("mean {} len: {}".format(z, masked.shape[0]))

        if masked.shape[0]==0:
            return None

        x = (point[0] - self.camera_intrinsic_["cx"]) * z  / self.camera_intrinsic_["fx"]
        y = (point[1] - self.camera_intrinsic_["cy"]) * z / self.camera_intrinsic_["fy"]
        return ((x, y, z))

    def detect_plane(self, person, image, depth):
        start = time.time()
        lines=[]
        poses = []
        for i in [2,1,5]:
            if person[i][0] > 0.0001 and person[i][1] > 0.0001:
                pos = self.calculate_depth(image, depth, (person[i][0], person[i][1]))
                if pos is not None:
                    poses.append(np.asarray(pos))

        if len(poses)<=1:
            rospy.logwarn("not enough point detected for shoulders")
            return
        if len(poses)>1:
            lines.append(poses[-1]-poses[0])

        for i in [8,9,10]:
            if person[i][0] > 0.0001 and person[i][1] > 0.0001:
                pos = self.calculate_depth(image, depth, (person[i][0], person[i][1]))
                if pos is not None:
                    lines.append(np.asarray(pos)-poses[0])
                    break

                print(point_x, point_y)
        if len(lines) != 2:
            rospy.loginfo("not enough lines detected, lines: {}".format(len(lines)))
            self.image_debug_pub_.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            return

        norm_plain = np.cross(lines[0], lines[1])
        rospy.loginfo("cross is: {}".format(norm_plain))

        if len(poses)>1:
            angle1 = math.atan2(norm_plain[2], norm_plain[0])
            angle2 = math.atan2(norm_plain[1], norm_plain[0])
            angle3 = math.atan2(norm_plain[2], norm_plain[1])
            cv2.putText(image, '{:2.1f}'.format(np.rad2deg(angle1)-90),(100, int(100)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0) , thickness=4, lineType=cv2.LINE_AA)
            cv2.putText(image, '{:2.2f}, {:2.2f}, {:2.2f}'.format(poses[0][0], poses[0][1], poses[0][2]),(100, int(200)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0) , thickness=4, lineType=cv2.LINE_AA)
            cv2.putText(image, '{:2.2f}, {:2.2f}, {:2.2f}'.format(poses[-1][0], poses[-1][1], poses[-1][2]),(100, int(300)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0) , thickness=4, lineType=cv2.LINE_AA)
            cv2.putText(image, '{:2.2f}, {:2.2f}, {:2.2f}'.format(pos[0], pos[1], pos[2]),(100, int(400)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0) , thickness=4, lineType=cv2.LINE_AA)

        if not args[0].no_display:
            self.image_debug_pub_.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

        end = time.time()
        print("detect plane finished. Total time: " + str(end - start) + " seconds")
        #except Exception as e:
        #    print(e)

    # points are four corner of the image coordinates
    def detect_lines(self, image, depth, point_coordinates):
        number_lines = 20
        lines = []
        for i in range(number_lines):
            line = []
            invalid = False
            for j in range (2):
                point_x = int(round(point_coordinates[j][0] + (point_coordinates[j+2][0]-point_coordinates[j][0])*float(i)/float(number_lines)))
                point_y = int(round(point_coordinates[j][1] + (point_coordinates[j+2][1]-point_coordinates[j][1])*float(i)/float(number_lines)))
                pos = self.calculate_depth(image, depth, (point_x, point_y))
                if pos is None:
                    invalid = True
                    break
                line.append(pos)
                cv2.circle(image, (point_x, point_y), 10, [0,255,0 ], 7)

            if invalid:
                continue
            lines.append(line)

        point_cloud = PointCloud()
        point_cloud.header.stamp = rospy.Time.now()
        point_cloud.header.frame_id = "zed2_left_camera_optical_frame"
        pos_person = []
        orientation_person = []
        base_line = lines[0]
        for line in lines[1:]:
            #m = -(line[1][0]-line[0][0])/(line[1][1] - line[0][1])

            person_vec_a = np.asarray(base_line[0]) - np.asarray(line[0])
            person_vec_b = np.asarray(line[1]) - np.asarray(line[0])
            prepen_person = np.cross(person_vec_a, person_vec_b)
            prepen_person_norm = prepen_person / np.linalg.norm(prepen_person)

            camera_vec = np.asarray([-1,0,0])
            angle = np.arccos(np.dot(prepen_person_norm,camera_vec ))
            orientation_person.append(angle)

            pos = (np.asarray(line[0]) + np.asarray(line[1]))/2
            pos_person.append(pos)
            for item in line:
                point = Point32()
                point.x = item[0]
                point.y = item[1]
                point.z = item[2]
                point_cloud.points.append(point)
        self.toros_pub_.publish(point_cloud)
        median_orientation = np.median(np.asarray(orientation_person))
        print ("pose: {} median: {}".format(orientation_person,median_orientation))
        self.publish_pose_person(pos_person[0], median_orientation)
        return lines



    def detect(self, image, depth):
        if self.camera_intrinsic_ is None:
            return
        start = time.time()
        # Process and display images
        datum = op.Datum()
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop([datum])
        end = time.time()
        print("detect successfully finished. Total time: " + str(end - start) + " seconds")
        return datum


if __name__=="__main__":
    rospy.init_node("personDetection")
    person_detection = PersonDetection(params)
    while not rospy.is_shutdown():
        rospy.spin()
