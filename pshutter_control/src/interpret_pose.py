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
        self.rate = rospy.Rate(0.25)

        self.sub = rospy.Subscriber('/body_tracking_data', MarkerArray, self.recieve_pose, queue_size=1)

        self.publish()

    def publish(self):

        while not rospy.is_shutdown():

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
        rs, ls, rw, lw, pv = markers[12], markers[5], markers[14], markers[7], markers[0]

        if rs.type != rw.type != pv.type != ls.type != lw.type: 
            print("Diff people")
            return

        #print(rs)
        #print(rw)
        #print(pv)

        # Horizontal Vectors
        rhvx, rhvy = rs.pose.position.x - pv.pose.position.x, rs.pose.position.y - pv.pose.position.y
        # Arm Vector
        ravx, ravy = rs.pose.position.x - rw.pose.position.x, rs.pose.position.y - rw.pose.position.y

        # Horizontal Vectors
        lhvx, lhvy = ls.pose.position.x - pv.pose.position.x, ls.pose.position.y - pv.pose.position.y
        # Arm Vector
        lavx, lavy = ls.pose.position.x - lw.pose.position.x, ls.pose.position.y - lw.pose.position.y

        rmag_a = math.sqrt(ravx ** 2 + ravy ** 2)
        rmag_h = math.sqrt(rhvx ** 2 + rhvy ** 2)

        lmag_a = math.sqrt(lavx ** 2 + lavy ** 2)
        lmag_h = math.sqrt(lhvx ** 2 + lhvy ** 2)

        rangle = np.arccos(np.dot((ravx, ravy), (rhvx, rhvy)) / (rmag_a * rmag_h))
        langle = np.arccos(np.dot((lavx, lavy), (lhvx, lhvy)) / (lmag_a * lmag_h))

        print(f"Right arm angle: {rangle}")
        print(f"Left  arm angle: {langle}")

        answer = "none"

        if langle >= 1:
            answer = "Zip"

        if rangle >= 1:
            answer = "Zap"

        if langle >= 1 and rangle >= 1.2:
            answer = "Boing"

        self.answer = answer

        print(f"Output: {answer}\n")


if __name__ == "__main__":
    poser = PoseInterpreter()