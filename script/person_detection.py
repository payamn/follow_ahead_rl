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

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()


class PersonDetection:
    def __init__(self, params_op):
        self.bridge = CvBridge()

        self.rgb_subb = Subscriber("/zed/zed_node/rgb/image_rect_color", Image)
        #self.rgb_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.camera_cb)
        self.depth_sub = Subscriber("/zed/zed_node/depth/depth_registered", Image)
        ats = ApproximateTimeSynchronizer([self.rgb_subb, self.depth_sub], queue_size=2, slop=2)
        ats.registerCallback(self.camera_cb)

        # starting openpose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params_op)
        self.opWrapper.start()


    def camera_cb(self, rgb_msg, depth_msg):
        if abs(rgb_msg.header.stamp.to_sec() - rospy.Time.now().to_sec()) > 0.1:
            #rospy.loginfo("skipping {}".format(rgb_msg.header.stamp.to_sec() - rospy.Time.now().to_sec()))
            return
        image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        self.detect(image, depth)
        #cv2.imshow("img", image)
        #cv2.waitKey(1)

    def detect(self, image, depth):
        #try:
        start = time.time()
        # Process and display images
        datum = op.Datum()
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop([datum])
        if datum.poseKeypoints.shape == ():
            return
        person = datum.poseKeypoints[0]
        poses = []
        for i in [1,2,5]:
            if person[i][0] > 0.0001 and person[i][1] > 0.0001:
                circle_img = np.zeros((depth.shape[0],depth.shape[1]), np.uint8)
                cv2.circle(image, (person[i][0], person[i][1]), 3, [0,255,0 ])
                cv2.circle(circle_img,(person[i][0], person[i][1]),6,255,-1)
                # masked = np.copy(depth)
                # masked =  cv2.bitwise_and(depth, masked, mask=circle_img)
                # masked = masked[masked!=float('inf')]
                # masked = masked[masked!=float('nan')]
                # masked = masked[masked>0]
                # #print (masked, masked.shape)
                # avg = np.average(masked)
                #rospy.loginfo("mean {}".format(avg))

                #datos = cv2.mean(depth, mask=circle_img)

                poses.append((person[i][0], person[i][1]))

        #print ("\nmeans: ")
        #for i,pos in enumerate(poses):
        #    print("{}: {:10.2}".format( i, pos[2]))
        if len(poses)>1:
            angle = math.atan2(poses[-1][1] - poses[0][1], poses[-1][0] - poses[0][0])
            cv2.putText(image, '{:10.1f}'.format(np.rad2deg(angle)),(int((poses[-1][0] + poses[0][0])/2), int((poses[-1][1] + poses[0][1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , thickness=2, lineType=cv2.LINE_AA)


        if not args[0].no_display:
            cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
            #cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", image)
            key = cv2.waitKey(1)

        end = time.time()
        print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
        #except Exception as e:
        #    print(e)


if __name__=="__main__":
    rospy.init_node("personDetection")
    person_detection = PersonDetection(params)
    while not rospy.is_shutdown():
        rospy.spin()
