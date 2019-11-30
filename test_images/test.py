import cv2
import numpy as np
#Read First Image
imgBGR=cv2.imread('test1.jpg')
#Read Second Image
imgRGB=cv2.imread('test2.jpg')

a = cv2.resize(imgBGR,480,320)
b = cv2.resize(imgRGB,480,320)

cv2.imshow('bgr',a)
cv2.imshow('rgb',b)
#concatanate image Horizontally
img_concate_Hori=np.concatenate((a,b),axis=1)
#concatanate image Vertically
img_concate_Verti=np.concatenate((a,b),axis=0)
cv2.imshow('concatenated_Hori',img_concate_Hori)
cv2.imshow('concatenated_Verti',img_concate_Verti)
cv2.waitKey(0)
cv2.destroyAllWindows()