import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the image from the file manager (update the path with your own image path)
image_path = 'C:\\sweathaswin\\m1.jpg'  # Update with your image path
image = cv2.imread(image_path)

# Convert the image to RGB (MediaPipe expects RGB format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to find pose landmarks
results = pose.process(image_rgb)

# Check if pose landmarks are detected (it will contain landmarks for all detected people)
if results.pose_landmarks:
    # In Multi-Person Pose Estimation, the pose landmarks are not in an iterable list, 
    # but you can access them directly from `results.pose_landmarks`
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Show the image with pose estimation
cv2.imshow('Pose Estimation - Multiple People', image)

# Wait for the user to press a key, then close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
