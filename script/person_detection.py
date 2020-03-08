# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time

import math
import numpy as np

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
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
params["model_folder"] = "/home/payam/Soft/openpose-1.5.1/models"
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

        self.rgb_subb = Subscriber("/zed/zed_node/rgb/image_rect_color", Image)
        #self.rgb_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.camera_cb)
        self.depth_sub = Subscriber("/zed/zed_node/depth/depth_registered", Image)
        ats = ApproximateTimeSynchronizer([self.rgb_subb, self.depth_sub], queue_size=2, slop=2)
        ats.registerCallback(self.camera_cb)

        self.camera_info_sub = rospy.Subscriber("/zed/zed_node/depth/camera_info", CameraInfo, self.camera_info_cb)

        self.image_debug_pub_ = rospy.Publisher("person_detection", Image)

        # starting openpose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params_op)
        self.opWrapper.start()

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


    def camera_cb(self, rgb_msg, depth_msg):
        if abs(rgb_msg.header.stamp.to_sec() - rospy.Time.now().to_sec()) > 0.1:
            #rospy.loginfo("skipping {}".format(rgb_msg.header.stamp.to_sec() - rospy.Time.now().to_sec()))
            return
        image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        self.detect(image, depth)
        #cv2.imshow("img", image)
        #cv2.waitKey(1)

    def calculate_depth(self, image, depth, point):
        circle_img = np.zeros((depth.shape[0],depth.shape[1]), np.uint8)
        cv2.circle(image, point, 10, [0,255,0 ], 7)
        cv2.circle(circle_img,(point[0], point[1]),10,255,-1)
        masked = np.copy(depth)
        masked =  cv2.bitwise_and(depth, masked, mask=circle_img)
        masked = masked[masked!=float('inf')]
        masked = masked[masked!=float('nan')]
        masked = masked[masked>0]

        z = np.median(masked)
        rospy.loginfo("mean {} len: {}".format(z, masked.shape[0]))

        if masked.shape[0]==0:
            return None

        x = (point[0] - self.camera_intrinsic_["cx"]) * z  / self.camera_intrinsic_["fx"]
        y = (point[1] - self.camera_intrinsic_["cy"]) * z / self.camera_intrinsic_["fy"]
        return ((x, y, z))


    def detect(self, image, depth):
        if self.camera_intrinsic_ is None:
            return
        start = time.time()
        # Process and display images
        datum = op.Datum()
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop([datum])
        if datum.poseKeypoints.shape == ():
            rospy.logwarn("open pose not detecting the person")
            return
        person = datum.poseKeypoints[0]
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

        if len(lines) != 2:
            rospy.loginfo("not enough lines detected, lines: {}".format(len(lines)))
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
            #cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
            # cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", image)
            # key = cv2.waitKey(1)
            self.image_debug_pub_.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

        end = time.time()
        print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
        #except Exception as e:
        #    print(e)


if __name__=="__main__":
    rospy.init_node("personDetection")
    person_detection = PersonDetection(params)
    while not rospy.is_shutdown():
        rospy.spin()
