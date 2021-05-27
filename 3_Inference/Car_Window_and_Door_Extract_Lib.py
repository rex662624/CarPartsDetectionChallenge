import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt

def process_the_image(original_image):
    #draw_image = original_image.copy()
    contour_img_car_window = preprocessing_the_car_window(original_image.copy())
    door_guided_image, window_contour_list, success_fail = draw_contour_of_the_car_window(contour_img_car_window)
    door_contour_list = []
    if(success_fail==0):
        door_contour_list = preprocessing_the_car_door(door_guided_image)

    return window_contour_list, door_contour_list



def preprocessing_the_car_door(door_guided_image):
    door_contour_list = []
    
    contours, hierarchy = cv.findContours(door_guided_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    area_threshold = 10000
    center_of_image = np.array((door_guided_image.shape[1]//2 ,door_guided_image.shape[0]//2))
    centerpoint_of_contour = []
    index_of_contour = []
    for i in range(len(contours)):  
        area = cv.contourArea(contours[i])
        if area>area_threshold:
            M = cv.moments(contours[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])        
            centerpoint_of_contour.append((cX, cY))
            index_of_contour.append(i)
    dist = np.linalg.norm(center_of_image - centerpoint_of_contour, axis=1)
    door_index = np.argmin(dist)
    #cv.drawContours(draw_image, contours, index_of_contour[door_index], (255,255,0), -1)
    #cv.circle(draw_image, center_of_image, 20, (255, 255, 255), -1)
    door_contour_list.append(contours[index_of_contour[door_index]])
    return door_contour_list

def preprocessing_the_car_window(image):
    image = cv.blur(image,(5,5))
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h = np.mean(image[:,:,0])
    s = np.mean(image[:,:,1])
    v = np.mean(image[:,:,2])
    #print("mean hsv", h, " ", s, " ",v)
    #image = cv.inRange(image, (15, 7, 0), (135, 204, 24))
    
    #=====深色車 淺色車分開處理
    if(v<100):#深色車
        image = cv.inRange(image, (20, 7, 0), (123, 153, 30))
    else:
        image = cv.inRange(image, (20, 7, 0), (123, 153, 50))
        
    #pathimage = cv.inRange(image, (26, 0, 0), (74, 255, 255))
    #image = window_door_image-pathimage
#     kernel = np.ones((23,23),np.uint8) 
#     image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)    

#     kernel = np.ones((5,5),np.uint8) 
#     image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    
    return image



#找到車窗線
def draw_contour_of_the_car_window(contour_img):
    
    window_contour_list = []
    
    door_guided_image = np.ones_like(contour_img, dtype=np.uint8)
    
    contours, hierarchy = cv.findContours(contour_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(original_img, contours, -1, (0,255,0), 3)
    area_threshold = 3000
    window_threshold = 10000
    window_extRight_list = []
    window_extLeft_list = []
    door_bottom = [[],2**32]#contour,Cx,Cy
    Left_Bottom_of_image = np.array([0, contour_img.shape[0]])
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        
        if area>area_threshold:
            # find the bottom point by x coordinate
#             contour_sorted = np.array(sorted(contours[i].reshape(contours[i].shape[0], contours[i].shape[2]) , key=lambda k: [k[0], k[1]]))
#             value, ind = np.unique(contour_sorted[:,0], return_index=True)
#             ind[0] = contour_sorted.shape[0]
#             res = contour_sorted[np.roll(ind, -1) -1]
            
#             bottom_contour = res.reshape(res.shape[0], 1, 2)
#             cv.drawContours(original_img, bottom_contour, -1, (255,255,0), 3)
            
            # 找到contour的中心點
            M = cv.moments(contours[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])        
            #print("center point", (cX, cY))
            
            #=====用中心點位置分辨車窗和車門
            if(cY<100):
                continue
                
            elif(cY>contour_img.shape[0]/2):#地板
                
                #和左下角的距離最小的那個就是
                boundRect = cv.boundingRect(contours[i])
                
                Left_Bottom = np.array([boundRect[0], boundRect[1]+boundRect[3]])#X,Y
                #print(Left_Bottom)
                dist = np.linalg.norm(Left_Bottom-Left_Bottom_of_image)
                #print(dist)
                if(dist < door_bottom[1]):#找y最大的(最下面的)
                    door_bottom = [contours[i], dist]
                    #print(extBot[1], " " ,extBot[0])
                #cv.circle(original_img, (Left_Bottom[0], Left_Bottom[1]), 20, (0, 255, 255), -1)
                #cv.drawContours(original_img, contours, i, (255,0,0), -1)
                #cv.drawContours(door_guided_image, contours, i, (0,0,0), -1)
                

            elif area>window_threshold:#車窗
                #cv.drawContours(original_img, contours, i, (0,0,255), -1)
                hull_list = []
                hull_list.append(cv.convexHull(contours[i]))
                
                #cv.drawContours(original_img, hull_list, 0, (0,0,255), -1)
                window_contour_list.append(hull_list[0])
                cv.drawContours(door_guided_image, hull_list, 0, (0, 0, 0), -1)
                
                #=====找車窗最右邊的點
                #x,y,w,h = cv.boundingRect(contours[i])
                #cv.rectangle(original_img, (x, y), (x + w, y + h), (0, 255,0), 2)
                extLeft = tuple(hull_list[-1][hull_list[-1][:, :, 0].argmin()][0])
                extRight = tuple(hull_list[-1][hull_list[-1][:, :, 0].argmax()][0])
                #車窗可能不只框到一個
                window_extLeft_list.append(extLeft)
                window_extRight_list.append(extRight)

    #沒找到窗戶或車門 直接return
    if(len(window_extRight_list) == 0 or len(door_bottom[0])==0):
        return np.ones_like(contour_img, dtype=np.uint8), window_contour_list, -1                
                
    #畫車門
    #cv.drawContours(original_img, door_bottom, 0, (255,0,0), -1)
    cv.drawContours(door_guided_image, door_bottom, 0, (0,0,0), -1)                

    
    #畫車窗最右邊的點
    extRight = max(window_extRight_list,key=lambda item:item[0])
    extLeft = min(window_extLeft_list,key=lambda item:item[0])
#     cv.circle(original_img, extRight, 20, (0, 255, 255), -1)
#     cv.circle(original_img, extLeft, 20, (0, 255, 255), -1)
        
    #cv.rectangle(original_img,(extRight[0], 0),(original_img.shape[1], original_img.shape[0]) , (0, 255,0), -1)    
    #cv.rectangle(original_img, (0, 0), (extLeft[0], original_img.shape[0]), (0, 255, 0), -1)
    cv.rectangle(door_guided_image, (extRight[0], 0),(contour_img.shape[1], contour_img.shape[0]), (0, 0, 0), -1)
    cv.rectangle(door_guided_image, (0, 0), (extLeft[0], contour_img.shape[0]), (0, 0, 0), -1)
    

    
    return door_guided_image, window_contour_list, 0


### Draw Window and Door


def draw_window_and_door(offset_x, offset_y, window_contour_list, door_contour_list, image):
    #print(len(window_contour_list), " ", len(door_contour_list))
    #print(np.array(window_contour_list[0]).shape)
    
    for i in range(len(window_contour_list)):
        cv.drawContours(image, window_contour_list, i, (0,0,255), -1, offset=(offset_x, offset_y)) 
        
    for i in range(len(door_contour_list)):
        cv.drawContours(image, door_contour_list, i, (255,0,0), -1, offset=(offset_x, offset_y)) 
    
    return image
