#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64MultiArray

class WakeupRobot:

    def __init__():

        # Publish to the physical robot
        pub = rospy.Publisher('/joint_group_controller/command', Float64MultiArray, queue_size=1)

        rospy.init_node('awaken', anonymous=True)
        timer = rospy.Rate(1)
            # pub = rospy.Publisher('/unity_joint_group_controller/command', Float64MultiArray)

        # Publish repeatedly
        i = 0;
        while not rospy.is_shutdown() and i < 20:
            joint_cmd = Float64MultiArray()
            # joint_cmd.data = [0, 0, 0, 0]
            joint_cmd.data = [0, -0.5, -0.2, -0.1]
            pub.publish(joint_cmd)
            timer.sleep()
            i += 1
        

if __name__ == '__main__':
    try:
        _ = WakeupRobot()
    except rospy.ROSInterruptException:
        pass