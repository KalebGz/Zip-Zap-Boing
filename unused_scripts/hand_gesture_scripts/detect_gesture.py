#!/usr/bin/env python

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import message_filters

import matplotlib.pyplot as plt
from PIL import Image as ImageShow
import os

import torch


# hand keypoint detector
from zzb_hand_tracker.hand_detector import HandDetector

# hand pose classifier
from zzb_hand_tracker.pose_classifier.train import ImageClassifier

# Hand Pose Classifier Model Path
HAND_POSE_MODEL_PARENT_PATH = '/home/{User}/catkin_ws/src/cpsc459-term-project/scripts/hand_tracker/classifier/weights/v1/' #TODO: Create scripts folder and add this


class DetectGesture():
    def __init__(self):

        # Init the node
        rospy.init_node('find_picked_object_node')

        self.hand_pose_model_name = 'epoch_20.pth'
        self.hand_pose_model_path = os.path.join(HAND_POSE_MODEL_PARENT_PATH, self.hand_pose_model_name)
        self.device = 'cuda'

        self.mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) #Recalculate
        self.hand_pose_classes = 3
        self.hand_pose_class_map = {0: "None", 1: "zip", 2:"zap", 3:"boing"}        
        
        self.image_counter = 0

        # Initialize models
        self.hand_detector = HandDetector()
        self.hand_pose_classifier.load_state_dict(torch.load(self.hand_pose_model_path))
        self.hand_pose_classifier = self.hand_pose_classifier.to(self.device)

        self.rgb_img_sub = message_filters.Subscriber('/camera3/color/image_raw/compressed', CompressedImage)

        # For Future Rosified version 
        # self.rgb_img_sub = message_filters.Subscriber('/camera2/color/image_raw', Image)
        # self.synchronizer = message_filters.TimeSynchronizer(
        # [self.rgb_img_sub], 10)
        # self.synchronizer.registerCallback(self.get_image_cb)

        # rospy.spin()



    def get_image_cb(self, ros_rgb_image):

        self.image_counter+=1

        # self.rgb_image = ros_numpy.numpify(ros_rgb_image)
        # self.rgb_image_timestamp = ros_rgb_image.header.stamp


        hand_detect_img = self.hand_detector.findHands(self.rgb_image) # viz
        lmlist = self.hand_detector.findPosition(hand_detect_img)
 
        # Pass img to run inference on the Pose classifier
        pick_pred = self.pick_classifier(hand_detect_img) # viz KK: integer here reprensents labels for pick classifier{0: "Zip", 1: "Zap", 2:"Boing"}        
        pick_pred = pick_pred.cpu().detach().numpy()
        pick_class = self.hand_pose_class_map[pick_pred[0]]


    def pick_classifier(self, image):
        '''
        Wrappper over the pick classifier. Manipulates the images in the same way as the train dataloader and 
        runs inference on the trained model

        Parameters:
                    image (image): Input image from get_image_cb
        Returns:
                    predicted (int): Predicted class 0/1/2 (None/PICK/NOT_PICK)
        '''
        device = self.device
        with torch.no_grad():
            image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_CUBIC)
            image = np.asarray(image, np.float32)
            image -= self.mean
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, axis=0)
            image = torch.tensor(image)
            outputs = self.hand_pose_classifier(image.to(device))
            outputs = outputs.to(device)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

    def visualize_pick_predictions(self, image, pick_pred):
        '''
        Visualize the predictions of the Pick classifier on an image

        Parameters:
                    image (image): Input image from get_image_cb
                    pick_pred (int): Predicted class 0/1/2 (None/PICK/NOT_PICK)
        Returns:
                    ann_image (image): Image with the prediction visualized on the top left corner of the image
        '''
        # ann_image = cv2.putText(image, pick_pred, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
        ann_image = image
        return ann_image
        # return image

    # def convert_ros_compressed_to_cv2(self, compressed_msg):
    # np_arr = np.fromstring(compressed_msg.data, np.uint8)
    # return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def cv2_to_imgmsg(self, cv_image):
        '''
        Helper function to publish a cv2 image as a ROS message (without using ROS cv2 Bridge)
        https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/

        Parameters:
                    cv_image (image): Image to publish to a ROS message
        Returns:
                    img_msg (message): Image message published to a topic 
        '''
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
        return img_msg

if __name__ == '__main__':
    try:
        node = DetectGesture()
    except rospy.ROSInterruptException:
        pass
