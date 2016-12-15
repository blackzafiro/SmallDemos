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

Y_SHIFT = 40
# The images have num_cluster colours
num_objects = 2
param_suits = {
    'sponge_set_1': {'file_video': 'data/generated_videos/sponge_centre_100__filterless_segmented.avi',
                     'color_indices':[0,1,2],   # color in pos 0 is background
                     'loss_threshold':25,
                     'median_kernel_size': 7,
                     'sobel_kernel_size': 3}
}

def get_objects(img, grays, loss_threshold):
    """
    Creates images with background in black and object in white
    :param img: presegmented image
    :param grays: list of grays off every object
    :return: list of images, each with an object in white
    """
    background_colour = grays[0]
    img_objects = []
    np.set_printoptions(threshold=np.inf)
    for i in range(1, len(grays)):
        # TODO: separate object
        binary = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        yes = np.logical_and(img >= grays[i] - loss_threshold,
                             img <= grays[i] + loss_threshold)[:,:,0]
        binary[yes] = 255
        binary[~yes] = 0
        img_objects.append(binary)
    return img_objects


def detect_borders(img, param_suit, num_label = 0):
    """ Detect borders using sobel. """
    ypos = num_label * (img.shape[0] + Y_SHIFT)
    # Apply median filtering
    cv2.imshow('Image ' + str(num_label), img)
    cv2.moveWindow('Image ' + str(num_label), 0, ypos)
    dst = cv2.medianBlur(img, param_suit['median_kernel_size'])
    cv2.imshow('Median filtered ' + str(num_label), dst)
    cv2.moveWindow('Median filtered ' + str(num_label), img.shape[1], ypos)

    # Use sobel edge detection
    ksize = param_suit['sobel_kernel_size']
    sobelx = np.absolute(cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=ksize))
    sobely = np.absolute(cv2.Sobel(dst, cv2.CV_64F, 0, 1, ksize=ksize))
    # Trick in http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    sobelx64f = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    # print(sobel_8u.shape)
    # print(sobel_8u.dtype)
    # print(type(sobel_8u[0]))
    # print(type(sobel_8u[0,0]))
    cv2.imshow('Sobel ' + str(num_label), sobel_8u)
    cv2.moveWindow('Sobel ' + str(num_label), 2 * dst.shape[1], ypos)

    return sobel_8u


def track_material(img, num_label = 0):
    """ Use NG to track deformable material. """
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    imc = np.zeros(im2.shape)
    #cnt = contours[0]
    #cv2.drawContours(im2, [cnt], 0, (0, 255, 0), 1)
    cv2.drawContours(imc, contours, -1, (0, 255, 0), 1)
    cv2.imshow('Contour', im2)
    cv2.moveWindow('Contour', 4 * img.shape[1], num_label * (img.shape[0] + Y_SHIFT))
    if cv2.waitKey() & 0xFF == ord('q'):
        sys.exit(-1)



def track_objects(contour_imgs):
    """ Function specific to the problem of tracking the finger and material
    pushed by the robot. """
    track_material(contour_imgs[0])


def track(cap, param_suit, process):
    """ Use NG to track contours"""
    grays = GNG.cluster_colours(num_objects + 1)[param_suit['color_indices']]
    loss_threshold = param_suit['loss_threshold']

    num_frame = 1
    while (cap.isOpened()):
        ## Capture frame-by-frame
        ret, frame = cap.read()
        if not ret: break

        ## Our operations on the frame come here
        start_time = time.time()

        images = get_objects(frame, grays, loss_threshold)
        dsts = []
        for i, img in enumerate(images):
            dsts.append(detect_borders(img, param_suit, i))

        process(dsts)

        print("--- %s seconds to process frame %d ---" % ((time.time() - start_time), num_frame))


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

    track(cap, param_suit, track_objects)

    print("Processing finished.  Press key to end program.")
    cv2.waitKey()
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
