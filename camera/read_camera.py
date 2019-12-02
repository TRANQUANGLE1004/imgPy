#system import
import numpy as np

import cv2

##########################[DEFINE]#################################
font = cv2.FONT_HERSHEY_PLAIN   

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

###################################################################
#                                                                 #                
#######################[custom function]###########################
#this function is a simple cv2.resizeWindow
def resizeWindow(window_name,val_with,val_height):
    return cv2.resizeWindow(window_name,val_with,val_height)

def imshow(window_name,img):
    return cv2.imshow(window_name,img)

def imread(path_file,flag):
    return imread(path_file,flag)

def waitKey(num):
    return cv2.waitKey(num)   

def resize(img,x,y):
    return cv2.resize(img,(x,y))

def moveWindow(window_name,x,y):
    return cv2.moveWindow(window_name,x,y)

def setWindowTitle(window_name,msg):
    return cv2.setWindowTitle(window_name,msg)

def show4Img(window_name,img_0,img_1,img_2,img_3):
    with_val = 720
    height_val = 540
    half_with = int(with_val/2)
    half_height = int(height_val/2)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    resizeWindow(window_name,with_val,height_val)
    #moveWindow(window_name,0,0) #move to home
    setWindowTitle(window_name,'hello world')

    #resize img
    img_0_fix = cv2.resize(img_0,(half_with,half_height))
    img_1_fix = cv2.resize(img_1,(half_with,half_height))
    img_2_fix = cv2.resize(img_2,(half_with,half_height))
    img_3_fix = cv2.resize(img_3,(half_with,half_height))

    row_0 = np.concatenate((img_0_fix,img_1_fix),axis=1)
    row_1 = np.concatenate((img_2_fix,img_3_fix),axis=1)
    pic = np.concatenate((row_0,row_1),axis=0)

    cv2.imshow(window_name,pic)

def convert2DTo3D(img):
    return cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

def inRange(img,low,high):
    return cv2.inRange(img,low,high)

def cvtColor(img,flag):
    return cv2.cvtColor(img,flag)
def GaussianBlur(img,kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

###################################################################

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.imread('../test_images/straight_lines2.jpg',1)

    # Our operations on the frame come here
    gray = cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    mask_white = inRange(gray,200,255)
    #mask_yellow = inRange(hsv,lower_yellow,upper_yellow)
    #mask_yw = cv2.bitwise_or(mask_white, mask_yellow)

    gauss_gray = GaussianBlur(mask_white,5)
    canny_egdes = cv2.Canny(gauss_gray,50,150)
    #gauss = cv2.GaussianBlur()    
    # Display the resulting frame
    #imshow('nomal',mask_white)

    

    show4Img('my output',frame,convert2DTo3D(gray),convert2DTo3D(mask_white),convert2DTo3D(canny_egdes))

    if waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()