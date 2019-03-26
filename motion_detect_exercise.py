# motion detection
import cv2
import numpy as np
import NameFind

video = cv2.VideoCapture("How People Walk.mp4")

#_, first_frame = video.read()
#first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
#first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)

Background_sub = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=40, detectShadows=False)


while True:
    _, frame = video.read()
   # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   # frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    #difference = cv2.absdiff(first_gray, frame_gray)
    #_, difference = cv2.threshold(difference, 45, 255, cv2.THRESH_BINARY)

    new_bs = Background_sub.apply(frame)

    #difference = cv2.dilate(new_bs, None, iterations=0)

    _, cnts, _ = cv2.findContours(new_bs.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < 2000:
            continue

        (x, y, w, h) = cv2.boundingRect(c)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)





    cv2.imshow('Frame', frame)
    cv2.imshow('Difference', new_bs)




    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()



