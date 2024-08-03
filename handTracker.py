import cv2
import mediapipe
import numpy as np
import pyautogui

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

initHand = mediapipe.solutions.hands  # Initializing mediapipe
# Object of mediapipe with "arguments for the hands module"
mainHand = initHand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
draw = mediapipe.solutions.drawing_utils  # Object to draw the connections between each finger index
wScr, hScr = pyautogui.size()  # Outputs the high and width of the screen (1920 x 1080)
pX, pY = 0, 0  # Previous x and y location
cX, cY = 0, 0  # Current x and y location
pyautogui.FAILSAFE = False

def handLandmarks(colorImg):
    landmarkList = []  # Default values if no landmarks are tracked

    landmarkPositions = mainHand.process(colorImg)  # Object for processing the video input
    landmarkCheck = landmarkPositions.multi_hand_landmarks  # Stores the out of the processing object (returns False on empty)
    if landmarkCheck:  # Checks if landmarks are tracked
        for hand in landmarkCheck:  # Landmarks for each hand
            for index, landmark in enumerate(hand.landmark):  # Loops through the 21 indexes and outputs their landmark coordinates (x, y, & z)
                draw.draw_landmarks(img, hand, initHand.HAND_CONNECTIONS)  # Draws each individual index on the hand with connections
                h, w, c = img.shape  # Height, width and channel on the image
                centerX, centerY = int(landmark.x * w), int(landmark.y * h)  # Converts the decimal coordinates relative to the image for each index
                landmarkList.append([index, centerX, centerY])  # Adding index and its coordinates to a list
                
    return landmarkList


def fingers(landmarks):
    fingerTips = []  # To store 4 sets of 1s or 0s
    tipIds = [4, 8, 12, 16, 20]  # Indexes for the tips of each finger
    
    # Check if thumb is up
    if landmarks[tipIds[0]][1] > landmarks[tipIds[0] - 1][1]:
        fingerTips.append(1)
    else:
        fingerTips.append(0)
    
    # Check if fingers are up except the thumb
    for id in range(1, 5):
        if landmarks[tipIds[id]][2] < landmarks[tipIds[id] - 3][2]:  # Checks to see if the tip of the finger is higher than the joint
            fingerTips.append(1)
        else:
            fingerTips.append(0)

    return fingerTips

def zoom(factor):
    pyautogui.keyDown('ctrl')
    pyautogui.scroll(factor)
    pyautogui.keyUp('ctrl')

def screenshot():
    pyautogui.screenshot("screenshot.png")

def scroll(direction):
    if direction == "up":
        pyautogui.scroll(1)
    elif direction == "down":
        pyautogui.scroll(-1)

while True:
    ret, img = cap.read()  # Reads frames from the camera
    if not ret:
        print("Error: Failed to capture frame.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Changes the format of the frames from BGR to RGB
    lmList = handLandmarks(imgRGB)
    # cv2.rectangle(img, (75, 75), (640 - 75, 480 - 75), (255, 0, 255), 2)
    
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Gets index 8s x and y values (skips index value because it starts from 1)
        x2, y2 = lmList[12][1:]  # Gets index 12s x and y values (skips index value because it starts from 1)
        finger = fingers(lmList)  # Calling the fingers function to check which fingers are up
        
        if finger[1] == 1 and finger[2] == 0:  # Checks to see if the pointing finger is up and thumb finger is down
            x3 = np.interp(x1, (75, 640 - 75), (0, wScr))  # Converts the width of the window relative to the screen width
            y3 = np.interp(y1, (75, 480 - 75), (0, hScr))  # Converts the height of the window relative to the screen height
            
            cX = pX + (x3 - pX) / 7  # Stores previous x locations to update current x location
            cY = pY + (y3 - pY) / 7  # Stores previous y locations to update current y location
            
            pyautogui.moveTo(wScr-cX, cY)  # Function to move the mouse to the x3 and y3 values (wSrc inverts the direction)
            pX, pY = cX, cY  # Stores the current x and y location as previous x and y location for next loop
        
        # Additional functionalities
        if finger[1] == 1 and finger[2] == 1:  # Checks if the index and thumb fingers are up
            pyautogui.dragTo(wScr-cX, cY)  # Function to move the mouse to the x3 and y3 values (wSrc inverts the direction)
            pX, pY = cX, cY  # Stores the current x and y location as previous x and y location for next loop

        if finger[0] == 1 and finger[1] == 0:  # Checks if the thumb is up and index finger is down
            pyautogui.click()  # Left click

        if finger[1] == 1 and finger[2] == 0 and finger[3] == 1 and finger[4] == 1:  # Checks if the thumb and index fingers are up
            zoom(1.5)  # Zoom in

        if finger[1] == 0 and finger[2] == 1 and finger[3] == 1 and finger[4] == 1:  # Checks if the thumb and index fingers are pinched
            zoom(-1.5)  # Zoom out

        if finger[1] == 0 and finger[2] == 0 and finger[3] == 1 and finger[4] == 1:  # Checks if the middle and ring fingers are up
            scroll("up")  # Scroll up

        if finger[1] == 1 and finger[2] == 1 and finger[3] == 0 and finger[4] == 0:  # Checks if the thumb and index fingers are down
            scroll("down")  # Scroll down

        if finger[0] == 1 and finger[1] == 0 and finger[2] == 0 and finger[3] == 0 and finger[4] == 0:  # Checks if only the thumb is up
            screenshot()  # Take a screenshot

    # Resize the image for display
    img = cv2.resize(img, (640, 480))  # Resizing the image to 640x480
    
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
