#!/usr/bin/python3.5

import sys
import os
OPENCV_HOME = os.environ['OPENCV_HOME']
sys.path.insert(1, OPENCV_HOME + '/lib/python3.5/dist-packages')

from pathlib import Path
import time
import pickle

import cv2
import numpy as np
import matplotlib.cm as cm

import GNG
from CvUtil import *

# The images have num_cluster colours
num_objects = 2
param_suits = {
    'sponge_set_1': {'file_video': 'data/generated_videos/sponge_centre_100__filterless_segmented.avi',
                     'color_indices':[0,1,2],   # color in pos 0 is background
                     'median_kernel_size': 7,
                     'sobel_kernel_size': 1}
}

def get_objects(img, grays):
    """
    Creates images with background in black and object in white
    :param img: presegmented image
    :param grays: list of grays off every object
    :return: list of images, each with an object in white
    """
    background_colour = grays[0]
    img_objects = []
    for i in range(1, len(grays)):
        # TODO: separate object
        img_objects.append(img)
    return img_objects


def detect_borders(img, param_suit, ypos = 0):
    """ Detect borders using sobel. """
    # Apply median filtering
    cv2.imshow('Image', img)
    cv2.moveWindow('Image', 0, ypos)
    dst = cv2.medianBlur(img, param_suit['median_kernel_size'])
    cv2.imshow('Median filtered', dst)
    cv2.moveWindow('Median filtered', img.shape[1], ypos)

    # Use sobel edge detection
    ksize = param_suit['sobel_kernel_size']
    sobelx = np.absolute(cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=ksize))
    sobely = np.absolute(cv2.Sobel(dst, cv2.CV_64F, 0, 1, ksize=ksize))
    sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    cv2.imshow('Sobel', sobel)
    cv2.moveWindow('Sobel', 2 * dst.shape[1], ypos)

    return dst


def track(cap, param_suit):
    """ Use NG to track contours"""
    grays = GNG.cluster_colours(num_objects + 1)[param_suit['color_indices']]

    num_frame = 1
    while (cap.isOpened()):
        ## Capture frame-by-frame
        ret, frame = cap.read()
        if not ret: break

        ## Our operations on the frame come here
        start_time = time.time()

        images = get_objects(frame, grays)
        for img in images:
            detect_borders(img, param_suit)

        print("--- %s seconds to process frame %d ---" % ((time.time() - start_time), num_frame))

        for i, img in enumerate(images):
            cv2.imshow('segmented'+str(i), frame)
            cv2.moveWindow('segmented'+str(i), i * img.shape[1], img.shape[0])

        print("Frame ", num_frame)
        num_frame += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


def print_usage(script_name):
    """Print script instructions."""
    print('Usage: ' + script_name + ' <param_set>')
    sys.exit(1)

if __name__ == '__main__':
    nargs = len(sys.argv)
    param_suit = None
    if(nargs == 2):
        if sys.argv[1] in param_suits:
            param_suit = param_suits[sys.argv[1]]
            cap = cv2.VideoCapture(param_suit['file_video'])
        else:
            print_usage(sys.argv[0])
    else:
        print_usage(sys.argv[0])

    track(cap, param_suit)

    print("Processing finished.  Press key to end program.")
    cv2.waitKey()
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
