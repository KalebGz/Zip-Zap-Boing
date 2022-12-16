import numpy as np 
from hand_detector import HandDetector
from realsensecv import RealsenseCapture
from PIL import Image as im
import cv2
import time
import itertools
import csv
import os

ROOT = os.path.expanduser("~")
FPS = 30
 
def record_data(detector, participant_id, participant_data_path, realsense, rgb_frames):
    ''' Opens camera stream, records video, detects hand keypoints and saves them '''
    master_kp_list = []

    pTime = 0
    cTime = 0

    if realsense:
        cap = RealsenseCapture()
        # Property setting
        cap.FPS = FPS
        # Unlike cv2.VideoCapture(), do not forget cap.start()
        cap.start()
        print("Cap started!")
    else:
        cap = cv2.VideoCapture(0)

    while True: # cap is Opened():
        success, frame = cap.read()

        if not realsense:
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if realsense:
            img = frame[0]
        else:
            img = frame


        # out_rgb.write(img)
        rgb_frames.append(img)
        # print(videos)
        
        if realsense:
            pass


        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        
        # Post-Process landmarks to save them in appropriate format          
        kp_list = postprocess_landmarks(lmlist)

        # Aggregate processed landmarks from all frames into a single list of lists
        master_kp_list.append(kp_list)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        if success:
            cv2.imshow("Image", img)

        key = cv2.waitKey(1) 

        # Press q to quit the video capture and save the output   
        if key == ord('q'):
            break
        
    # Store output keypoints in CSV file
    store_csv_file(master_kp_list, participant_id, participant_data_path)  

    # Close the window / Release webcam
    cap.release()

    
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows() 

    return 
        
def postprocess_landmarks(hand_landmarks): 
    ''' Takes all the hand landmarks and reformats them as one array to make it easy to organize in a csv'''
    tuple_list = []
    for idx, kp in enumerate(hand_landmarks):
        x = kp[1]
        y = kp[2]
        xy_tuple = (x,y)
        tuple_list.append(xy_tuple) # [(x0, y0), (x1, y1), ....]
    xy_list = list(itertools.chain(*tuple_list)) # [x0, y0, x1, y1, ....]
    return xy_list

def store_csv_file(kp_list, participant_id, participant_data_path):
    ''' Save keypoints into CSV file '''
    csv_filename = "P_" + participant_id + '.csv'
    csv_path = os.path.join(participant_data_path, csv_filename)
    with open(csv_path, 'w') as f:
        write = csv.writer(f)
        write.writerows(kp_list)
        print("Done Writing CSV!")        

def write_video(rgb_frames, participant_id, participant_data_path):
    ''' Save processed frames into video for labeling'''
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    channel = 3
    video_filename = "P_" + participant_id + '.webm'
    video_path = os.path.join(participant_data_path, video_filename)

    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    out_rgb = cv2.VideoWriter(video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    print("LENGTH OF FRAMES {}".format(len(rgb_frames)))
    for i, frame in enumerate(rgb_frames):        
        out_rgb.write(frame)     
    print("DONE WRITTING RGB VIDEO!") 


def main():
    rgb_frames = [] 
    realsense = False
    detector = HandDetector()
    participant_id = input("Enter participant ID: ")
    participant_data_path = os.path.join(ROOT, "hand_classification_data", "P_"+participant_id)
    if not os.path.exists(participant_data_path):
        os.makedirs(participant_data_path)
    record_data(detector, participant_id, participant_data_path, realsense, rgb_frames )
    write_video(rgb_frames, participant_id, participant_data_path)
    


if __name__ == "__main__":
    main()