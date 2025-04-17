import cv2
# Load an input image
image = cv2.imread(r"C:\sweathaswin\s.jpg") # Replace with your image path
if image is None:
 print(" Could not load image. Please check the file path.")
 exit()
# 1. Line
line_img = image.copy()
cv2.line(line_img, (50, 50), (300, 50), (0, 255, 0), 3)
cv2.imshow("Line", line_img)
# 2. Rectangle
rect_img = image.copy()
cv2.rectangle(rect_img, (100, 100), (400, 200), (255, 0, 0), 3)
cv2.imshow("Rectangle", rect_img)
# 3. Circle
circle_img = image.copy()
cv2.circle(circle_img, (250, 250), 50, (0, 0, 255), -1)
cv2.imshow("Circle", circle_img)
# 4. Ellipse
ellipse_img = image.copy()
cv2.ellipse(ellipse_img, (300, 150), (100, 40), 0, 0, 360, (0, 255, 255), 2)
cv2.imshow("Ellipse", ellipse_img)
# 5. Text
text_img = image.copy()
cv2.putText(text_img, "Experiment No 2 is Done!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
(255, 255, 255), 2)
cv2.imshow("Text", text_img)
