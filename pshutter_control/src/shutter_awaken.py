#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def wakeup_robot():

    pub = rospy.Publisher('/unity_joint_group_controller/command', String)
    # pub = rospy.Publisher('/joint_group_controller/command', String)

    rospy.init_node('awaken!!', anonymous=True)

    while not rospy.is_shutdown():
        joint_cmd = 
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        break;

if __name__ == '__main__':
    try:
        wakeup_robot()
    except rospy.ROSInterruptException:
        pass