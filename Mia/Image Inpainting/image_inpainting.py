import numpy as np
import cv2 as cv
img = cv.imread('Project-Computer-Vision\\Mia\\Image Inpainting\\messi.png')
mask = cv.imread('Project-Computer-Vision\\Mia\\Image Inpainting\\mask.png', cv.IMREAD_GRAYSCALE)

# Resize the images to have the same dimensions
img1 = cv.resize(img, (640, 480))  
mask1 = cv.resize(mask, (640, 480)) 

dst = cv.inpaint(img1,mask1,3,cv.INPAINT_TELEA)

cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()