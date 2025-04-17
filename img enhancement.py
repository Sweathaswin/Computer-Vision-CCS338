import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread(r"C:\sweathaswin\s.jpg")  # Replace with your image path
if image is None:
    print("Image not found. Please check the file path.")
    exit()

# Helper function to label images
def label_image(img, text):
    labeled = img.copy()
    cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled

# 1. Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", label_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Grayscale"))

# 2. HSV Conversion
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", label_image(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "HSV Color Space"))

# 3. Histogram Equalization (on grayscale)
plt.figure(figsize=(6, 4))
plt.title("Original Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.hist(gray.ravel(), bins=256, range=(0, 256), color='gray')
plt.tight_layout()
plt.show()

equalized = cv2.equalizeHist(gray)
cv2.imshow("Histogram Equalization", label_image(cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR), "Histogram Equalization"))

# 4. Image Smoothing
avg_blur = cv2.blur(image, (5, 5))
cv2.imshow("Average Blur", label_image(avg_blur, "Average Blur"))

gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Gaussian Blur", label_image(gaussian_blur, "Gaussian Blur"))

median_blur = cv2.medianBlur(image, 5)
cv2.imshow("Median Blur", label_image(median_blur, "Median Blur"))

# 5. Gradient Operations
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

cv2.imshow("Sobel X", label_image(cv2.convertScaleAbs(sobel_x), "Sobel X"))
cv2.imshow("Sobel Y", label_image(cv2.convertScaleAbs(sobel_y), "Sobel Y"))
cv2.imshow("Sobel Combined", label_image(cv2.convertScaleAbs(sobel_combined), "Sobel Combined"))

laplacian = cv2.Laplacian(gray, cv2.CV_64F)
cv2.imshow("Laplacian", label_image(cv2.convertScaleAbs(laplacian), "Laplacian"))

# 6. Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)
cv2.imshow("Canny Edges", label_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "Canny Edge Detection"))

cv2.waitKey(0)
cv2.destroyAllWindows()
