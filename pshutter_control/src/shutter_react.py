#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float64MultiArray

class MoveReaction:

    def __init__(self):

        self.cmd_mappings = {"Zap": [-0.5, -0.5, -0.2, -0.1],
                             "Zip": [0.5, -0.5, -0.2, -0.1],
                             "Boing": [0, -0.5, -0.2, 0.6]}

        sub = rospy.Subscriber('/zzb_move_robot', String, self.update_pose, queue_size=1)
        self.pub = rospy.Publisher('/joint_group_controller/command', Float64MultiArray, queue_size=1)

        rospy.init_node('shutter_react', anonymous=True)
        self.timer = rospy.Rate(1)

        rospy.spin()

    def update_pose(self, msg):

        # print(msg)
        
        joint_cmd = Float64MultiArray()
        joint_cmd.data = self.cmd_mappings[msg.data]

        # print(joint_cmd)

        self.pub.publish(joint_cmd)
        self.timer.sleep()


if __name__ == '__main__':
    try:
        _ = MoveReaction()
    except rospy.ROSInterruptException:
        pass