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
# inspection_pts = { 
#                     'front_window': [[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],
#                     'back_window':  [[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],            
#                     'front_door':   [[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],
#                     'back_door':    [[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],
#                  }



#image = Image.open('00001.jpg')
#prediction, new_image = yolo.detect_image(image)

cap = cv2.VideoCapture('Point\\Sample1.mp4')
#cap = cv2.VideoCapture('Production_Line.mp4')
#cap.set(cv2.CAP_PROP_POS_FRAMES, 12000)
#cap.set(cv2.CAP_PROP_POS_FRAMES, 5000)
frame_count = 0
#前輪出現時間 後輪出現時間 大小車 後輪時間-前輪時間
#511 1488 大車(Sample2) 977
#466 1714 小車(Sample1) 1248
#314 1474 大車(Sample3) 1120
#329 1523 大車(Sample4) 1194
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
#cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
time_start =  time.time()

Prev_Window_List, Prev_Door_List = [], []
window_contour_list, door_contour_list = [], []

prev_centerX, prev_centerY = 0, 0
Wheel_Record = []

Acc_Wheel = 0
#Golden = np.load('Golden.npy')
Golden = np.load('Golden.npy')

Golden_Pt = Golden[:,:-1]
#[True, center_x, center_y, frame_count, index_min, Nowx, Nowy]
Find_Front = [False,2**32-1,2**32-1,-1,-1,-1,-1]
Find_Back = [False,2**32-1,2**32-1,-1,-1,-1,-1]
print(Golden_Pt)
missing = 0
Find_Wheelx, Find_Wheely = -1, -1
initilize = 0
prev_frame_time, new_frame_time = 0, 0
Upper = 300

x_min, y_min = 0, 0
x_max, y_max = 0, 0
inspection_v = 0

def bottom_line(x):
    return int(-((650-x)*400/650-680))
    

#當前輪x>多少的時候，這個點會開始出現 [x, v,(點)] v是速度
#[triggerx, triggery, velocity, accelerate,(x,y),frame累積, GoldenFrame]
#479 546
# 0, 0.11, 20, 0.3
# Inspection_pt = [
#                 #[triggerx, triggery,(x,y),frame累積, True False, velocity, accelerate, GoldenFrame]
#                 [553, 600, (630, 500), 0, False, 0, 0.01], [553, 600, (630, 550), 0, False, 0, 0.01], [553, 600, (630, 600), 0, False, 0, 0.01],
#                 [479, 546, (780, 550), 0, False, 0, 0.3], [479, 546, (780, 600), 0, False, 0, 0.3], [479, 546, (780, 650), 0, False, 0, 0.3],
#                 [474, 565, (780, 550), 0, False, 0, 0.11], [345, 486, (740, 520), 0, False, 0, 0.11],
#                 [412, 527, (740, 400), 0, False, 0, 0.11], [412, 527, (680, 300), 0, False, 0, 0.11]
#                 ]

# valid col [already insepction, point position]
Inspection_pt = [
                [False,[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],
                [False,[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],
                [False,[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],
                [False,[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],
                [False,[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],
                [False,[0,(0,0)],[0,(0,0)],[0,(0,0)],[0,(0,0)]],
    
                ]  
line_list = [[],[],[],[],[],[]]     
Interval = 110
IntervalAcc = 0    
ColCount = 0             
#====
Golden_Pt = np.array(Golden_Pt)
#====
for i in range(len(Inspection_pt)):
    #dist = np.linalg.norm([Inspection_pt[i][2][0]]-Golden_Pt[:,0], axis=0)
    dist = abs(Inspection_pt[i][2][0]-Golden_Pt[:,0])
    # print(Golden_Pt[:,0])
    # print(Inspection_pt[i][2][0])
    # print(dist)
    index_min = np.argmin(dist)
    Inspection_pt[i].append(index_min)
    #print(Golden_Pt[index_min])
# print(Inspection_pt)

#exit(0)



Inspection_pt_offset_x, Inspection_pt_offset_y = 0, 0

while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read() 

    # cv2.imwrite('test\\temp2.jpg', frame) 
    # exit(0)

    frame_count += 1
    # if(frame_count%100==0):
    #     cv2.imwrite('Test\\frame{}.jpg'.format(frame_count), frame)
    if(Find_Front[0]==False):
        offsetx = 2*(frame.shape[0])//3
        offsety = 2*(frame.shape[1])//3  
    # Use the Right left
    # if(not ret):
    #     np.save('table.npy', np.array(Wheel_Record))
    right_bottom = frame.copy()[offsety:, offsetx:]
    # 顯示圖片
    prediction, new_image = yolo.detect_image(Image.fromarray(cv2.cvtColor(right_bottom, cv2.COLOR_BGR2RGB)))

    new_image = frame
    #new_image = cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)
    
    time_now = time.time() - time_start
    second = time_now
    #print(prediction)
    #print(second)

    Door_List = []
    Wheel_List = []  
    
    #cv2.imwrite('temp\\test.jpg', frame)    
    #break
    # output as xmin, ymin, xmax, ymax, class_index, confidence
    now_centerX, now_centerY = 0, 0

    if(len(prediction)==0):
        missing+=1
    for object_detected in prediction:
        object_name = class_list[object_detected[4]]
            
        # if(object_name == 'Door'):
        #     #cv2.rectangle(new_image, (object_detected[0], object_detected[1]), (object_detected[2], object_detected[3]), (255, 0, 0), 2)
        #     x_width = (object_detected[2] - object_detected[0])
        #     y_width = (object_detected[3] - object_detected[1])

        #     Door = np.array(frame[object_detected[1]:object_detected[3],object_detected[0]:object_detected[2]])
        #     Door_List.append(object_detected)
        #     #Door = cv2.blur(Door,(11,11))
        #     #Door = cv2.cvtColor(Door, cv2.COLOR_BGR2GRAY)
        #     #_, Gray = cv2.threshold(Door, 20, 255, cv2.THRESH_BINARY_INV)

        #     #cv2.imshow('Gray', Gray)     

        #     #cv2.imwrite('temp\Door{}.jpg'.format(second), Door)     
        # elif(object_name == 'SideGlass'):
        #     #cv2.rectangle(new_image, (object_detected[0], object_detected[1]), (object_detected[2], object_detected[3]), (0, 255, 0), 2)
        #     center_x = (object_detected[0] + object_detected[2])//2
        #     center_y = (object_detected[1] + object_detected[3])//2
        #     #cv2.circle(new_image, (center_x, center_y),10 , (0, 0, 255), -1)            
        
        if(object_name == 'Wheel'):

            center_x = (object_detected[0] + object_detected[2])//2 + offsetx  
            center_y = (object_detected[1] + object_detected[3])//2 + offsety
            Wheel_List.append([object_detected, center_x, center_y])
            #cv2.circle(new_image, (center_x, center_y), 10 ,(0, 0, 255), -1)
            
            #cv2.rectangle(new_image, (object_detected[0], object_detected[1]), (object_detected[2], object_detected[3]), (0, 0, 255), 2)
            if(((abs(center_x-prev_centerX)<30 and abs(center_y-prev_centerY)<30) or prev_centerX==0) and Acc_Wheel<10):

                if(Find_Front[0] == False):#應該畫面內只會有前輪，因此找到的一定是前輪
                    Acc_Wheel+=1
                    prev_centerX, prev_centerY = center_x, center_y
                    #cv2.circle(new_image, (center_x, center_y), 10 ,(255, 0, 0), -1)

                elif(abs(center_x-Find_Wheelx)>200):#要找後輪，但可能前後輪都被偵測到，要排除前輪
                    Acc_Wheel+=1
                    prev_centerX, prev_centerY = center_x, center_y
                    #cv2.circle(new_image, (center_x, center_y), 10 ,(0, 255, 0), -1)

                else:
                    missing+=1

            elif(Acc_Wheel>=10):
                dist = np.linalg.norm([prev_centerX, prev_centerY]-Golden_Pt, axis=1)
                index_min = np.argmin(dist)
                
                if(Find_Front[0] == False):
                    Find_Front = [True, center_x, center_y, frame_count, index_min, center_x, center_y]#offsetx offsety
                    prev_centerX, prev_centerY, missing, Acc_Wheel = 0, 0, 0, 0
                    print(frame_count)
                    
                    #======檢測點
                    line_list[0] = [[center_x, Upper_int],[center_x, center_y]]
                    Inspection_pt[0][0] = True
                    ColCount+=1

                elif(Find_Back[0] == False):
                    if(np.min(dist)>50):
                        prev_centerX, prev_centerY, missing, Acc_Wheel = 0, 0, 0, 0
                    else:#找到後輪
                        Find_Back = [True, center_x, center_y, frame_count, index_min, center_x, center_y]#offsetx offsety
                        #print(frame_count)
                        
                    #cv2.circle(new_image, (prev_centerX, prev_centerY), 10 ,(0, 255, 255), -1)
                #print(Golden_Pt[index_min])
                #cv2.circle(new_image, (prev_centerX, prev_centerY), 10 ,(0, 255, 255), -1)
                #print(prev_centerX, prev_centerY)

            else:
                missing+=1
        else:
            missing+=1

    #print(missing)
    if(missing>10):
        prev_centerX, prev_centerY, missing, Acc_Wheel = 0, 0, 0, 0

    
    Wheel_Pt = []
    Wheel_Find_List = [Find_Front, Find_Back]
    for i in range(len(Wheel_Find_List)):
        Find_Wheel = Wheel_Find_List[i]

        if(Find_Wheel[0] == True):
            GoldenX, GoldenY, Golden_Start_Frame = Golden[Find_Wheel[4]][0], Golden[Find_Wheel[4]][1], Golden[Find_Wheel[4]][2]
            # print(GoldenX, GoldenY)
            # exit(0)

            Start_Frame = Find_Wheel[3]
            if((Find_Wheel[4]+frame_count-Start_Frame) < len(Golden)):
                Wheeloffsetx = Golden[Find_Wheel[4]+(frame_count-Start_Frame)][0] - GoldenX
                Wheeloffsety = Golden[Find_Wheel[4]+(frame_count-Start_Frame)][1] - GoldenY
                
                Find_Wheelx, Find_Wheely = Find_Wheel[1]+Wheeloffsetx, Find_Wheel[2]+Wheeloffsety
                
                Find_Wheel[5], Find_Wheel[6] = Find_Wheelx, Find_Wheely
                #cv2.circle(new_image, (Golden_Pt[Find_Wheel[4]][0]+Wheeloffsetx, Golden_Pt[Find_Wheel[4]][1]+Wheeloffsety), 10 ,(0, 0, 0), -1)
                #cv2.circle(new_image, (Find_Wheelx, Find_Wheely), 10 ,(255, 255, 255), -1)
                #if(frame_count==1200):
                # if(Find_Wheelx<=412):
                #     print(Find_Wheelx, Find_Wheely)
                #     cv2.imwrite('Test\\tempfronthand.jpg', frame)    
                #     exit(0)
                #存前後輪座標
                Wheel_Pt.append([Find_Wheelx, Find_Wheely])


    Upper-=0.025
    Upper_int = int(Upper)

    Door = np.array([])

    if(len(Wheel_Pt)==0):# not find the wheel
        #print("Not find the wheel")
        pass
    elif(len(Wheel_Pt)==1):# not find the wheel
        #print("find front wheel")
        #cv2.rectangle(new_image, (Wheel_Pt[0][0], Upper_int), (frame.shape[1], frame.shape[0]), (255, 0, 0), 2)
        
        Door = np.array(frame[Upper_int:frame.shape[0],Wheel_Pt[0][0]:frame.shape[1]])

        #用現在的xmin-前一個x_min
        if(x_min!=0):
            inspection_v =Wheel_Pt[0][0]- x_min
        #inspection_v-=0.05
        y_min, y_max = Upper_int, frame.shape[0]
        x_min, x_max = Wheel_Pt[0][0], frame.shape[1]
        
    elif(len(Wheel_Pt)==2):# find the both wheel
        #print("find both wheel")
        #cv2.rectangle(new_image, (Wheel_Pt[0][0], Upper_int), (Wheel_Pt[1][0], Wheel_Pt[1][1]), (255, 0, 0), 2)
        Door = np.array(frame[Upper_int:Wheel_Pt[1][1], Wheel_Pt[0][0]:Wheel_Pt[1][0]])
        #用現在的xmin-前一個x_min
        if(x_min!=0):
            inspection_v =Wheel_Pt[0][0]- x_min
        #inspection_v-= 0.2
        y_min, y_max = Upper_int, Wheel_Pt[1][1]
        x_min, x_max = Wheel_Pt[0][0], Wheel_Pt[1][0]



    #=======================================畫檢測點===========================================
    if(Wheel_Find_List[1][0]==False):#還沒找到後輪
        # 77    # 136    # 201    # 252    # 299    # 334    # 362    # 393            
        #先看看觸發新的點col了沒
        #Now position - Start Position
        if((Find_Front[1]-Find_Front[5]) >IntervalAcc and (Find_Front[1]-Find_Front[5])<300 and ColCount<6):
            Interval -=15         
            IntervalAcc+=Interval
            #print("===",IntervalAcc)

            #===先算初始位置
            #x = int(x_max-((x_max-x_min)/(ColCount)))
            line_list[ColCount] = [[760, y_min],[760, y_max]]
            #=============
            Inspection_pt[ColCount][0] = True
            ColCount+=1
            
        #print(Inspection_pt[0])
        #print(line_list[0])
        #算出新的col位置
        for i in range(len(Inspection_pt)):
            
            if(Inspection_pt[i][0]==True):
                line_list[i][0][0] = line_list[i][1][0] =  int(line_list[i][0][0] + inspection_v)
                line_list[i][1][1] = bottom_line(line_list[i][1][0])
                if(i<3):
                    line_list[i][0][1] = line_list[i][1][1]-150-90*i#line_list[i][1][1]-150-100*i
                    # if(i==2):
                    #     print(line_list[i][0][1])
                    #     exit(0)
                else:
                    line_list[i][0][1] = 300#280
                #print(line_list[i])
                #cv2.line(new_image, tuple(line_list[i][0]),tuple(line_list[i][1]), (0,250,250),3)
                # 整條線去算

                #for j in range(1,):

    else:#找到後輪了 用九等分演算法
        inspection_temp_inteval = (x_max-x_min)/6
        for i in range(len(Inspection_pt)):
            tempx = int(x_min+i*inspection_temp_inteval)
            line_list[i][0][0] = line_list[i][1][0] = tempx
            line_list[i][1][1] = bottom_line(tempx)
            if(i==0):
                line_list[i][0][1] = line_list[i][1][1]-150
            else:
                line_list[i][0][1] = y_min
            #cv2.line(new_image, tuple(line_list[i][0]),tuple(line_list[i][1]), (0,250,250),3)
        

    #畫圖
    for i in range(len(Inspection_pt)):
        if(Inspection_pt[i][0]==True):
            number_of_pt = len(Inspection_pt[i])-1
            interval_now = (line_list[i][1][1]-line_list[i][0][1])/number_of_pt

            for j in range(len(Inspection_pt[i])-1):
                cv2.circle(new_image, (line_list[i][0][0], int(line_list[i][0][1]+interval_now*j)), 4 ,(150, 150, 150), -1)
            #cv2.line(new_image, tuple(),tuple(), (0,250,250),3)


    cv2.imshow('frame', new_image)



    #print(Wheel_Record)
    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #np.save('table.npy', np.array(Wheel_Record))
        break

