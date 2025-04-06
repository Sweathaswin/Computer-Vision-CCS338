import cvzone
import cv2
import pyautogui  # For simulating keyboard input
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# Gesture-to-Action Mapping
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand1 = hands[0]  # First hand detected
        fingers = detector.fingersUp(hand1)

        # Gesture Detection
        if fingers == [1, 0, 0, 0, 0]:
            print("Move Left")
            pyautogui.press('left')

        elif fingers == [0, 0, 0, 0, 1]:
            print("Move Right")
            pyautogui.press('right')

        elif fingers == [1, 1, 1, 1, 1]:
            print("Jump")
            pyautogui.press('up')

        elif fingers == [0, 0, 0, 0, 0]:
            print("Skid")
            pyautogui.press('down')

    # Display the webcam feed
    cv2.imshow("Image", img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()