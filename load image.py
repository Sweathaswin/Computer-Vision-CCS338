import cv2

# Load the image (replace 'image_path' with your image file path)
image = cv2.imread('C:\sweathaswin')

# Check if the image was loaded correctly
if image is None:
    print("Error: Unable to load image")
else:
    # Display the image
    cv2.imshow('Image', image)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
