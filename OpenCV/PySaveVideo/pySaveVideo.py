#!/usr/bin/python3

import sys
import os
OPENCV_HOME = os.environ['OPENCV_HOME']
sys.path.append(OPENCV_HOME + '/lib/python3.5/dist-packages')

import numpy as np
import cv2

def test_opencv():
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'PIM1')
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_opencv()
