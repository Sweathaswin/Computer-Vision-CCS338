import numpy as np
import cv2

def triangulate_points(proj_matrix1, proj_matrix2, points1, points2):
    """
    Function to triangulate 3D points from 2D corresponding points in two views.
    Arguments:
    - proj_matrix1: Projection matrix of the first camera.
    - proj_matrix2: Projection matrix of the second camera.
    - points1: Corresponding 2D points in the first image.
    - points2: Corresponding 2D points in the second image.

    Returns:
    - 3D points in homogeneous coordinates.
    """
    # Convert points to homogeneous coordinates
    points1_homogeneous = cv2.convertPointsToHomogeneous(points1)
    points2_homogeneous = cv2.convertPointsToHomogeneous(points2)

    # Triangulate points
    points_3d_homogeneous = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1_homogeneous, points2_homogeneous)

    # Convert points from homogeneous coordinates to 3D (x, y, z)
    points_3d = points_3d_homogeneous[:3] / points_3d_homogeneous[3]
    
    return points_3d.T

# Camera calibration data (Example values, you should replace these with your actual calibration data)
K = np.array([[1000, 0, 320],  # fx, 0, cx
              [0, 1000, 240],  # 0, fy, cy
              [0, 0, 1]])  # 0, 0, 1

R1 = np.eye(3)  # Camera 1 rotation matrix (identity matrix for simplicity)
t1 = np.array([0, 0, 0]).reshape(3, 1)  # Camera 1 translation vector

R2 = np.eye(3)  # Camera 2 rotation matrix (identity matrix for simplicity)
t2 = np.array([1, 0, 0]).reshape(3, 1)  # Camera 2 translation vector (camera is shifted by 1 unit along the x-axis)

# Projection matrices for both cameras
proj_matrix1 = np.dot(K, np.hstack((R1, t1)))  # P1 = K [R1 | t1]
proj_matrix2 = np.dot(K, np.hstack((R2, t2)))  # P2 = K [R2 | t2]

# Function to find matching keypoints using ORB
def find_matching_keypoints(image1, image2):
    orb = cv2.ORB_create()

    # Detect ORB keypoints and descriptors in both images
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # Match descriptors using the brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort the matches based on distance (best matches first)
    matches = sorted(matches, key = lambda x:x.distance)

    # Extract the matching keypoints
    points1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
    
    return points1, points2

# Load images from file system
image1_path = 'C:\sweathaswin\q'  # Update with your actual image path for the first image
image2_path = 'C:\sweathaswin\q2'  # Update with your actual image path for the second image

image1 = cv2.imread(image1_path)  # Read first image
image2 = cv2.imread(image2_path)  # Read second image

# Convert images to grayscale (ORB works better with grayscale images)
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Find matching keypoints between the two images
points1, points2 = find_matching_keypoints(image1_gray, image2_gray)

# Triangulate the points
points_3d = triangulate_points(proj_matrix1, proj_matrix2, points1, points2)

# Print the 3D points
print("3D Points:")
print(points_3d)

# Optionally, display the 2D points in both images (for visualization)
for p1, p2 in zip(points1, points2):
    cv2.circle(image1, (int(p1[0]), int(p1[1])), 5, (0, 255, 0), -1)
    cv2.circle(image2, (int(p2[0]), int(p2[1])), 5, (0, 255, 0), -1)

# Show the images with matching points
cv2.imshow("Image 1 with keypoints", image1)
cv2.imshow("Image 2 with keypoints", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
