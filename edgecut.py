import cv2
import numpy as np
# Load the image
img =cv2.imread(r"C:\sweathaswin\s.jpg")
if img is None:
 print("Image not found. Please check the path.")
 exit()
# Create an initial mask
mask = np.zeros(img.shape[:2], np.uint8)
# Define models for background and foreground
bgModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)
# Define a rectangle around the object (x, y, width, height)
rect = (50, 50, 300, 300)
# Apply GrabCut algorithm
cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
# Modify the mask: 0 & 2 => background, 1 & 3 => foreground
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
result = img * mask2[:, :, np.newaxis]
# Display output
cv2.imshow("Original Image", img)
cv2.imshow("Segmented Foreground", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
