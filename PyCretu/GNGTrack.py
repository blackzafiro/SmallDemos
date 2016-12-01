#!/usr/bin/python3.5

import sys
import os
OPENCV_HOME = os.environ['OPENCV_HOME']
sys.path.append(OPENCV_HOME + '/lib/python3.5/dist-packages')

from pathlib import Path
import time
import pickle

import cv2
import math
import numpy as np
import matplotlib.cm as cm

# http://scikit-learn.org/stable/install.html
# sudo -H pip3 install -U scikit-learn
from sklearn import cluster
import GNG
from CvUtil import *


segment_params = {'max_age': 40,
                  'lambda_steps': 1000,        # insert node evey lambda steps
                  'epsilon_beta': 0.05,        # 0 < beta < 1
                  'epsilon_eta': 0.0006,       # 0 < eta < 1
                  'alfa': 0.5,                 # 0 < alfa < 1
                  'beta': 0.0005               # 0 < beta < 1
                  }

vision_params = {'median_kernel_size': 7}

sponge_set_1 = {'file_video': '/home/blackzafiro/Programacion/MassSpringIV/data/press/sponge_centre_100.mp4',
                'roi': ((325, 200), (925, 700)),
                'file_segment_gng': 'luv_5charac_segment_gng_scentre.pickle', }


## Could try
## http://docs.opencv.org/3.1.0/df/d9d/tutorial_py_colorspaces.html
## as well


### Using only one color channel

def getSegmentationGNG(file_segment_gng = None, plot = False):
    """ Load GNG from file or create it. """
    gng = None
    if file_segment_gng:
        seg_path = Path(file_segment_gng)
        if seg_path.is_file():
            with seg_path.open(mode='rb') as f:
                gng = pickle.load(f)
                if hasattr(gng, 'fig'):
                    del gng.fig
                if hasattr(gng, 'ax'):
                    del gng.ax
    if gng is None:
        start_time = time.time()
        gng = GNG.calibrateSegmentationGNG(luv, segment_params)
        print("--- %s seconds to callibrate segmentation GNG ---" % (time.time() - start_time))
        if param_suit:
            with open(param_suit['file_segment_gng'], 'wb') as f:
                pickle.dump(gng, f)
    if plot:
        gng.show()
        gng.plotNetColorNodes(with_edges=True)
    return gng


def detect_borders(img, ypos = 0):
    """ Detect borders using sobel. """
    # Apply median filtering
    cv2.imshow('Image', img)
    cv2.moveWindow('Image', 0, ypos)
    dst = cv2.medianBlur(img, vision_params['median_kernel_size'])
    cv2.imshow('Median filtered', dst)
    cv2.moveWindow('Median filtered', dst.shape[1], ypos)

    # Use sobel edge detection
    sobelx = np.absolute(cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=1))
    sobely = np.absolute(cv2.Sobel(dst, cv2.CV_64F, 0, 1, ksize=1))
    sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    cv2.imshow('Sobel', sobel)
    cv2.moveWindow('Sobel', 2 * dst.shape[1], ypos)

    return dst

if __name__ == '__main__':
    nargs = len(sys.argv)
    param_suit = None
    if(nargs == 1):
        cap = cv2.VideoCapture(0)
    elif(nargs == 2):
        if sys.argv[1] == 'sponge_set_1':
            param_suit = sponge_set_1
            cap = cv2.VideoCapture(param_suit['file_video'])
        else:
            cap = cv2.VideoCapture(sys.argv[1])
    else:
        print('Usage: ' + sys.argv[0] + ' <video_file_name>')
        sys.exit(1)

    #
    # Process first frame
    #

    ret, frame = None, None
    if cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture first frame")
            sys.exit(2)
    else:
        print("Failed to capture source")
        sys.exit(1)

    if param_suit:
        roi_coords = param_suit['roi']
        small_frame = frame[roi_coords[0][1]:roi_coords[1][1], roi_coords[0][0]:roi_coords[1][0]]
    else:
        small_frame = frame

    small_frame = cv2.resize(small_frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
    showChannels(small_frame, ypos=small_frame.shape[0], wait=True)

    luv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2Luv)
    showChannels(luv, ypos=small_frame.shape[0], wait=True)
    #luv = small_frame

    if param_suit:
        roi_coords = param_suit['roi']
        cv2.rectangle(frame, roi_coords[0], roi_coords[1], 255)
        cv2.imshow('Source', frame)
        if cv2.waitKey() & 0xFF == ord('q'):
            sys.exit(0)
        else:
            cv2.destroyWindow('Source')

    # Display hsv frame
    cv2.imshow('rgb', small_frame)
    cv2.moveWindow('rgb', 0, 0)
    cv2.imshow('Color space', luv)
    cv2.moveWindow('Color space', 300, 0)
    if cv2.waitKey() & 0xFF == ord('q'):
        sys.exit(0)

    segmentGNG = getSegmentationGNG(param_suit['file_segment_gng'] if param_suit else None,
                                    False if param_suit else True)

    # Use K-Means to separate foreground from background
    dst = np.zeros((luv.shape[0], luv.shape[1]), dtype=np.uint8)

    if not hasattr(segmentGNG, 'kmeans'):
        print("Clustering neurons...")
        segmentGNG.extract_clusters(gng_node_indexes=[1])
        segmentGNG.show()
        with open(param_suit['file_segment_gng'], 'wb') as f:
            pickle.dump(segmentGNG, f)
    else:
        segmentGNG.show()
        segmentGNG.plotNetColorNodes(with_edges=True)

    print("Press key to segment first image...")
    if cv2.waitKey() & 0xFF == ord('q'):
        sys.exit(0)
    print("Segmenting first image...")
    segmentGNG.segment_image(luv, dst)
    dst = detect_borders(dst)

    print("Press key to start tracking")
    if cv2.waitKey() & 0xFF == ord('q'):
        sys.exit(0)

    cv2.destroyAllWindows()
    print("Start tracking...")

    num_frame = 2
    while(cap.isOpened()):
        ## Capture frame-by-frame
        ret, frame = cap.read()
        if not ret: break

        ## Our operations on the frame come here
        start_time = time.time()

        if param_suit:
            small_frame = frame[roi_coords[0][1]:roi_coords[1][1], roi_coords[0][0]:roi_coords[1][0]]
        else:
            small_frame = frame
        small_frame = cv2.resize(small_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        luv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2Luv)
        segmentGNG.segment_image(luv, dst)
        dst = detect_borders(dst)

        print("--- %s seconds to process frame %d ---" % ((time.time() - start_time), num_frame))
        ## Display the resulting frame
        cv2.imshow('luv', luv)
        cv2.moveWindow('luv', 3 * dst.shape[1], 0)

        print("Frame ", num_frame)
        num_frame += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    if cv2.waitKey() & 0xFF == ord('q'):
        sys.exit(0)

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
