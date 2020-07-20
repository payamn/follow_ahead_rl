
import rospy
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.msg import ModelStates
import tf2_ros


class TF_State_Publisher():
    def __init__(self):
        print("init")
        self.model_states_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_cb)
        self.transform_broadcaster_ = tf2_ros.TransformBroadcaster()

    def model_states_cb(self,  states_msg):
        for model_idx in range(len(states_msg.name)):
            if "tb3" in states_msg.name[model_idx]:
                pos = states_msg.pose[model_idx]
                state = {}
                self.publish_tf(pos, "{}/base_link".format(states_msg.name[model_idx]), "{}/odom".format(states_msg.name[model_idx]))


    def publish_tf(self, pose, child, parent):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation.x = pose.orientation.x
        t.transform.rotation.y = pose.orientation.y
        t.transform.rotation.z = pose.orientation.z
        t.transform.rotation.w = pose.orientation.w

        self.transform_broadcaster_.sendTransform(t)

if __name__=="__main__":
    rospy.init_node("tf_state_publisher_person")
    state_publisher = TF_State_Publisher()
    while not rospy.is_shutdown():
        rospy.spin()

