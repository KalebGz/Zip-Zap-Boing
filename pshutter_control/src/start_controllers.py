#!/usr/bin/env python

import rospy
from controller_manager_msgs.srv import SwitchController

def call_service():

    rospy.init_node("start_controllers", anonymous=True)

    rospy.wait_for_service('/controller_manager/switch_controller')
    try:
        args = SwitchController()
        args.start_controllers = ['joint_group_controller']
        args.strictness = 1

        service = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        response = service(start_controllers=args.start_controllers, strictness=args.strictness)
        return response
    except rospy.ServiceException as e:
        return "Service call failed!"

if __name__ == "__main__":
    print(call_service())