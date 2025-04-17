import cv2

image_path = r"C:\sweathaswin\i3.jpg"
image = cv2.imread(image_path)

# Check if image is loaded correctly
if image is None:
    print("Error: Image not found!")
    exit()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set a threshold value
threshold_value = 128

# Apply thresholding
_, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

# Display the original and thresholded images
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', thresholded_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
