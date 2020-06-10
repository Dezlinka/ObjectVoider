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
    colored = cv2.bitwise_and(frame, blur, mask=thresh)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        cv2.drawContours(frame, [c], -1, (255, 0, 0), 2, cv2.LINE_AA, hierarchy, 0)
    cv2.imshow('Frame with contours and KP', frame_with_kp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
