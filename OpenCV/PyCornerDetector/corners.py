#!/usr/bin/python3.5

# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

import sys
import os
OPENCV_HOME = os.environ['OPENCV_HOME']
sys.path.append(OPENCV_HOME + '/lib/python3.5/dist-packages')

import cv2
import numpy as np


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
        ret, img = cap.read()
        if not ret: break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        # src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst>0.01*dst.max()]=[0,0,255]

        cv2.imshow('dst',img)
        if cv2.waitKey(0) & 0xff == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


