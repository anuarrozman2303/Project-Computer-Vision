import numpy as np
import cv2

# Define the 3D object points (coordinates of points on calibration pattern)
object_points = []
for i in range(6):
    for j in range(8):
        object_points.append([j, i, 0]) # assuming z=0 as the calibration pattern lies on a plane

object_points = np.array(object_points, dtype=np.float32)

# Initialize arrays to store object points and image points from multiple images
object_points_list = []
image_points_list = []

# Load and process multiple images for calibration
for i in range(1, 11):  # assuming 10 calibration images named as 'calibration1.jpg', 'calibration2.jpg', ...
    image = cv2.imread("calib_radial.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect chessboard corners in the image
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    if ret:  # if corners are detected
        object_points_list.append(object_points)
        image_points_list.append(corners)

# Perform camera calibration
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points_list, image_points_list,
                                                                          gray.shape[::-1], None, None)

# Get optimal new camera matrix
img = cv2.imread('calib_radial.jpg')
h, w = img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))

# Compute undistortion maps
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffs, None, new_camera_matrix, (w, h), 5)

# Undistort the example image
undistorted_image = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

np.savez('B.npz', mtx=camera_matrix, dist=distortion_coeffs, rvecs=rvecs, tvecs=tvecs)

# Display original and undistorted image side by side
cv2.imshow("Original Image", img)
cv2.imshow("Undistorted Image", undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


