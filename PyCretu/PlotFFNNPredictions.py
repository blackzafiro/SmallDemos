#!/usr/bin/python3.5

"""
Draw FeedForward Neural Network Predictions over original videos.
"""

import sys
import os
OPENCV_HOME = os.environ['OPENCV_HOME']
sys.path.insert(1, OPENCV_HOME + '/lib/python3.5/dist-packages')
import cv2

import numpy as np

from Util import print_msg
from FFPredict import load_data, param_suits as ff_param_suits

param_suits = {
    'sponge_set_1': {
        'file_predictions': 'data/pickles/sponge_center_predictions.csv',
        'file_video': 'data/generated_videos/sponge_centre_100__filterless_segmented.avi',
        'file_results': 'data/generated_videos/sponge_centre_100__result.avi',
        'roi_shape': (600, 500, 3),
        'scale': 0.5
    }
}


def plot(param_name, param_suit):
    """
    Draw FeedForward Neural Network Predictions over original videos.
    """
    contours = np.loadtxt(param_suit['file_predictions'])
    print_msg(type(contours), contours.dtype, contours.shape)
    contours = contours.astype(np.int32)
    print_msg(type(contours), contours.dtype, contours.shape)
    contours = contours.reshape((contours.shape[0], int(contours.shape[1]/2), 2))
    print_msg(type(contours), contours.dtype, contours.shape)

    ff_param_suit = ff_param_suits[param_name]
    X, Y = load_data(ff_param_suit['file_train_data'],
                     ff_param_suit['file_force_data'])
    Y = Y.astype(np.int32)
    Y = Y.reshape((Y.shape[0], int(Y.shape[1] / 2), 2))

    cap = cv2.VideoCapture(param_suit['file_video'])
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(param_suit['file_results'], fourcc, 30, (300, 250))

    i = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, img = cap.read()
        if not ret: break

        #img = np.zeros((param_suit['roi_shape']))
        cv2.polylines(img, [contours[i]], True, 200, 2)
        cv2.polylines(img, [Y[i]], True, (0, 200, 0), 2)
        cv2.imshow('Demo', img)
        out.write(img)

        i += 1
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    out.release()
    cv2.waitKey(-1)


def print_instructions():
    """ Prints usage instructions. """
    print("Use: " + sys.argv[0] + " <set_of_paramteres>\nOptions:")
    for key in param_suits.keys():
        print('\t', key)


if __name__ == '__main__':
    nargs = len(sys.argv)
    if nargs != 2 or sys.argv[1] not in param_suits:
        print_instructions()
        sys.exit(1)
    param_suit = param_suits[sys.argv[1]]

    plot(sys.argv[1], param_suit)
