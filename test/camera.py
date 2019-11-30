import numpy as np
import cv2
import sys
import math



lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

#cap = cv2.VideoCapture(0)

frame = cv2.imread('../test_images/straight_lines2.jpg',1)

# Our operations on the frame come here
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#TODO:

mask_yellow = cv2.inRange(hsv_img,lower_yellow,upper_yellow)
mask_white = cv2.inRange(gray, 200, 255)
mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
mask_yw_image = cv2.bitwise_and(gray, mask_yw)
gauss_img = cv2.GaussianBlur(mask_yw_image,(5,5),0)

low_threshold = 50
high_threshold = 150
canny_edges = canny(gauss_img,low_threshold,high_threshold)

# Resize img
frameResize = cv2.resize(frame,(480,320))
grayResize = cv2.resize(gray,(480,320))
hsvResize = cv2.resize(hsv_img,(480,320))
mask_white_Resize = cv2.resize(mask_white,(480,320))
mask_yw_Resize = cv2.resize(mask_yw,(480,320))

mask_yellow_Resize = cv2.resize(mask_yellow,(480,320))
# 

# img_concate_Hori = np.concatenate((frameResize,hsvResize),axis=1)
# img_concate_Veti = np.concatenate((mask_yellow_Resize,mask_yellow_Resize),axis=1)
# img_concate = np.concatenate((img_concate_Hori,img_concate_Veti),axis=0)
cv2.imshow('nomal ',frameResize)
cv2.imshow('gray',grayResize)
cv2.imshow('canny',cv2.resize(canny_edges,(480,320)))
cv2.imshow('gauss',cv2.resize(gauss_img,(480,320)))

# Display the resulting frame


# move window
    

cv2.waitKey(0)
cv2.destroyAllWindows()

