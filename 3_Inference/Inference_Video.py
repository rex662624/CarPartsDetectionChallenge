import os
import sys
import Car_Window_and_Door_Extract_Lib

def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)

import argparse 
from keras_yolo3.yolo import YOLO, detect_video
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object
import test
import utils
import pandas as pd
import numpy as np
from Get_File_Paths import GetFileList
import random

import time

image_test_folder = ''
model_weights = 'Model_Weights\\trained_weights_final.h5'
classes_path = 'Model_Weights\data_classes.txt'
anchors_path = 'D:\Chenger\Car\CarPartsDetectionChallenge\\2_Training\src\keras_yolo3\model_data\yolo_anchors.txt'
gpu_num = 1
score = 0.25

import cv2
class_list = []
with open(classes_path) as f:
    class_list = f.readlines()

class_list = [x.replace('\n','') for x in class_list]

print(class_list)
yolo = YOLO(
    **{
        "model_path": model_weights,
        "anchors_path": anchors_path,
        "classes_path": classes_path,
        "score": score,
        "gpu_num": gpu_num,
        "model_image_size": (416, 416),
    }
)

#
inspection_pts = { 
                    'front_window': [[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],
                    'back_window':  [[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],            
                    'front_door':   [[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],
                    'back_door':    [[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],
                 }



#image = Image.open('00001.jpg')
#prediction, new_image = yolo.detect_image(image)

cap = cv2.VideoCapture('Car.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
time_start =  time.time()

while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    # 顯示圖片
    prediction, new_image = yolo.detect_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    new_image = cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)
    
    time_now = time.time() - time_start
    second = time_now
    #print(prediction)
    #print(second)

    Door_List = []

    # output as xmin, ymin, xmax, ymax, class_index, confidence
    for object_detected in prediction:
        object_name = class_list[object_detected[4]]
            
        if(object_name == 'Door'):
            cv2.rectangle(new_image, (object_detected[0], object_detected[1]), (object_detected[2], object_detected[3]), (255, 0, 0), 2)
            x_width = (object_detected[2] - object_detected[0])
            y_width = (object_detected[3] - object_detected[1])

            Door = np.array(frame[object_detected[1]:object_detected[3],object_detected[0]:object_detected[2]])
            Door_List.append(object_detected)
            #Door = cv2.blur(Door,(11,11))
            #Door = cv2.cvtColor(Door, cv2.COLOR_BGR2GRAY)
            #_, Gray = cv2.threshold(Door, 20, 255, cv2.THRESH_BINARY_INV)

            #cv2.imshow('Gray', Gray)     

            #cv2.imwrite('temp\Door{}.jpg'.format(second), Door)     

        elif(object_name == 'Wheel'):
            cv2.rectangle(new_image, (object_detected[0], object_detected[1]), (object_detected[2], object_detected[3]), (0, 0, 255), 2)
            center_x = (object_detected[0] + object_detected[2])//2
            center_y = (object_detected[1] + object_detected[3])//2
            cv2.circle(new_image, (center_x, center_y),10 , (0, 0, 255), -1)

        elif(object_name == 'SideGlass'):
            cv2.rectangle(new_image, (object_detected[0], object_detected[1]), (object_detected[2], object_detected[3]), (0, 255, 0), 2)
            center_x = (object_detected[0] + object_detected[2])//2
            center_y = (object_detected[1] + object_detected[3])//2
            cv2.circle(new_image, (center_x, center_y),10 , (0, 0, 255), -1)            

    # if find some door
    if(len(Door_List)):
        x_min = min(Door_List, key=lambda x: x[0])[0]
        y_min = min(Door_List, key=lambda x: x[1])[1]
        x_max = max(Door_List, key=lambda x: x[2])[2]
        y_max = max(Door_List, key=lambda x: x[3])[3]
        #print(x_min," ", y_min, "", x_max," ", y_max)
        cv2.rectangle(new_image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 5)
        Door = np.array(frame[y_min:y_max,x_min:x_max])
        cv2.imwrite('temp\Door{}.jpg'.format(second), Door)
            

    cv2.imshow('frame', new_image)
    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

