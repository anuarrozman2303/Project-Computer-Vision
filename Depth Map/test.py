import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img_left = cv.imread('Project-Computer-Vision\\Depth Map\\1.png', 0)
img_right = cv.imread('Project-Computer-Vision\\Depth Map\\2.png', 0)

# Resize the images to have the same dimensions
img_left = cv.resize(img_left, (640, 480))  
img_right = cv.resize(img_right, (640, 480)) 
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(img_left,img_right)
plt.imshow(disparity,'gray')
plt.show()