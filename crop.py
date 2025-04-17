import cv2

image_path = r"C:\sweathaswin\i2.jpg"
image = cv2.imread(image_path)

# Check if the image was loaded properly
if image is None:
    print("Error: Unable to load image.")
else:
    # Specify the coordinates of the region of interest (ROI)
    x1, y1 = 100, 50  # Top-left corner
    x2, y2 = 300, 200  # Bottom-right corner

    # Crop the image
    cropped_image = image[y1:y2, x1:x2]

    # Display the original image
    cv2.imshow('Original Image', image)

    # Display the cropped image
    cv2.imshow('Cropped Image', cropped_image)

    # Wait until a key is pressed
    cv2.waitKey(0)

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
