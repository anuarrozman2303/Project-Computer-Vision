import cv2

img_left = cv2.imread('Project-Computer-Vision\\Depth Map\\1.png', 0)
img_right = cv2.imread('Project-Computer-Vision\\Depth Map\\2.png', 0)

# Resize the images to have the same dimensions
img_left = cv2.resize(img_left, (640, 480))  
img_right = cv2.resize(img_right, (640, 480)) 

# Create a stereoBM object
stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)

# Compute the disparity map
disparity = stereo.compute(img_left, img_right)

# Normalize the disparity map for display
disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display the original left and right images, and the resulting disparity map
cv2.imshow('Left Image', img_left)
cv2.imshow('Right Image', img_right)
cv2.imshow('Disparity Map', disparity_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()