#!/usr/bin/env python

import rospy
import math
import numpy as np

from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

class PoseInterpreter:

    def __init__(self):

        self.answer = None

        rospy.init_node('pose_interpreter', anonymous=False)

        self.pub = rospy.Publisher('/zzb_move', String, queue_size=1)
        self.rate = rospy.Rate(0.5)

        self.sub = rospy.Subscriber('/body_tracking_data', MarkerArray, self.recieve_pose, queue_size=1)

        self.publish()

    def publish(self):

        while not rospy.is_shutdown():

            print(self.answer)

            if self.answer and self.answer != "none":
                print(f"publishing! {self.answer}")
                self.pub.publish(self.answer)
                pass

            self.rate.sleep()


    def recieve_pose(self, msg):

        markers = msg.markers
        answer = String

        if not len(msg.markers): return

        # Retrieve only one person
        rs, rw, pv = markers[12], markers[14], markers[0]

        if rs.type != rw.type != pv.type: 
            print("Diff people")
            return

        #print(rs)
        #print(rw)
        #print(pv)

        # Horizontal Vector
        hvx, hvy = rs.pose.position.x - pv.pose.position.x, rs.pose.position.y - pv.pose.position.y

        # Arm Vector
        avx, avy = rs.pose.position.x - rw.pose.position.x, rs.pose.position.y - rw.pose.position.y

        mag_a = math.sqrt(avx ** 2 + avy ** 2)
        mag_h = math.sqrt(hvx ** 2 + hvy ** 2)

        angle = np.arccos(np.dot((avx, avy), (hvx, hvy)) / (mag_a * mag_h))
        print(angle)

        if 0.6 < angle <= 1.4:
            answer = "Zip"
        elif 1.4 < angle <= 2:
            answer = "Zap"
        elif 2 < angle <= 3.14:
            answer = "Boing"
        else:
            answer = "none"

        self.answer = answer
        print(answer)


if __name__ == "__main__":
    poser = PoseInterpreter()