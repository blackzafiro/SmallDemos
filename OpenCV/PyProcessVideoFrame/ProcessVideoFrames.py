#!/usr/bin/python3.5

import sys
import os
OPENCV_HOME = os.environ['OPENCV_HOME']
#sys.path.append('/home/blackzafiro/Descargas/Programacion/opencv/opencv-3.1.0/build/lib/python3.5/dist-packages')
sys.path.append(OPENCV_HOME + '/lib/python3.5/dist-packages')

import numpy as np
import cv2

if __name__ == '__main__':
    nargs = len(sys.argv)
    if(nargs == 1):
        cap = cv2.VideoCapture(0)
    elif(nargs == 2):
        cap = cv2.VideoCapture(sys.argv[1])
    else:
        print('Usage: ' + sys.argv[0] + ' <video_file_name>')
        sys.exit(1)

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret: break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
