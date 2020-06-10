import cv2
import numpy as np
from cv2 import VideoCapture
def callback(x):pass
cap = VideoCapture(0)
cv2.namedWindow('image')
ilowH = 0
ihighH = 179
ilowS = 0
ihighS = 255
ilowV = 0
ihighV = 255
cv2.createTrackbar('lowH','image',ilowH,179,callback)
cv2.createTrackbar('highH','image',ihighH,179,callback)
cv2.createTrackbar('lowS','image',ilowS,255,callback)
cv2.createTrackbar('highS','image',ihighS,255,callback)
cv2.createTrackbar('lowV','image',ilowV,255,callback)
cv2.createTrackbar('highV','image',ihighV,255,callback)
detector = cv2.SimpleBlobDetector_create()
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')
    hsv_min = np.array([ilowH, ilowS, ilowV])
    hsv_max = np.array([ihighH, ihighS, ihighV])
    blur = cv2.GaussianBlur(frame, (5, 5), 4)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    keypoints = detector.detect(thresh)
    frame_with_kp = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Frame with contours and KP', frame_with_kp)
    cv2.imshow('thresh', thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

