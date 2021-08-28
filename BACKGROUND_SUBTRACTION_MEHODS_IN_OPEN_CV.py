import numpy as np
import cv2
cap = cv2.VideoCapture('vtest.avi')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
# fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG MASK Frame', fgmask)

    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()