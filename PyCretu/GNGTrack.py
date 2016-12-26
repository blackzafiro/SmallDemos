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
import NG

from CvUtil import *

Y_SHIFT = 150
# The images have num_cluster colours
num_objects = 2

## Parameters for GNG Segmentation Network
tracking_params = {'px_cicles': 8,
                  'max_age': 40,
                  'lambda_steps': 100,        # insert node evey lambda steps
                  'epsilon_beta': 0.05,        # 0 < beta < 1
                  'epsilon_eta': 0.0006,       # 0 < eta < 1
                  'alfa': 0.5,                 # 0 < alfa < 1
                  'beta': 0.0005               # 0 < beta < 1
                  }
neural_gas_params = {'epsilon': 0.05,
                     'plambda': 1,
                     'data_cicles': 20}

param_suits = {
    'sponge_set_1': {'file_video': 'data/generated_videos/sponge_centre_100__filterless_segmented.avi',
                     'file_save': 'data/pickles/sponge_set_1_track',
                     'train': False,
                     'file_initial_contour':'data/pickles/sponge_initial_contour.csv',
                     'color_indices': [0,1,2],   # color in pos 0 is background
                     'loss_threshold': 25,
                     'median_kernel_size': 7,
                     'sobel_kernel_size': 1},
    'sponge_set_2': {'file_video': 'data/generated_videos/sponge_longside_100__filterless_segmented.avi',
                     'file_save': 'data/pickles/sponge_set_2_track',
                     'train': False,
                     'file_initial_contour':'data/pickles/sponge_initial_contour.csv',
                     'color_indices': [0,1,2],   # color in pos 0 is background
                     'loss_threshold': 25,
                     'median_kernel_size': 7,
                     'sobel_kernel_size': 1
                     },
    'sponge_set_3': {'file_video': 'data/generated_videos/sponge_shortside_100__filterless_segmented.avi',
                     'file_save': 'data/pickles/sponge_set_3_track',
                     'train': False,
                     'file_initial_contour':'data/pickles/sponge_initial_contour.csv',
                     'color_indices': [0,1,2],   # color in pos 0 is background
                     'loss_threshold': 25,
                     'median_kernel_size': 7,
                     'sobel_kernel_size': 1
                     },
    'plasticine_set_1': {'file_video': 'data/generated_videos/a_plasticine_centre_100__filterless_segmented.avi',
                         'file_save': 'data/pickles/plasticine_set_1_track',
                         'train': False,
                         'file_initial_contour':'data/pickles/plasticine_initial_contour.csv',
                         'color_indices': [1,0,2],   # color in pos 0 is background
                         'loss_threshold': 25,
                         'median_kernel_size': 9,
                         'sobel_kernel_size': 1},
    'plasticine_set_2': {'file_video': 'data/generated_videos/a_plasticine_longside_100__filterless_segmented.avi',
                         'file_save': 'data/pickles/plasticine_set_2_track',
                         'train': False,
                         'file_initial_contour':'data/pickles/plasticine_initial_contour.csv',
                         'color_indices': [1,0,2],   # color in pos 0 is background
                         'loss_threshold': 25,
                         'median_kernel_size': 9,
                         'sobel_kernel_size': 1},
    'plasticine_set_3': {'file_video': 'data/generated_videos/a_plasticine_shortside_100__filterless_segmented.avi',
                         'file_save': 'data/pickles/plasticine_set_3_track',
                         'train': False,
                         'file_initial_contour':'data/pickles/plasticine_initial_contour.csv',
                         'color_indices': [1,0,2],   # color in pos 0 is background
                         'loss_threshold': 25,
                         'median_kernel_size': 9,
                         'sobel_kernel_size': 1}
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


class FingerMaterialTracker:
    """ Tracks the finger and the material using GNG and NG networks. """
    material_num_label = 0
    finger_num_label = 1

    def __init__(self, contour_imgs, file_initial_contour, train=True):
        """ Initializes tracker data. """
        finger_image = contour_imgs[self.finger_num_label]
        self._track_finger(finger_image)
        cv2.moveWindow('Contour ' + str(self.finger_num_label), 3 * finger_image.shape[1], self.finger_num_label * (finger_image.shape[0] + Y_SHIFT))

        self.contour = self._init_material(contour_imgs[self.material_num_label], file_initial_contour, train)

        self.finger_positions = [self.finger_position.copy()]
        self.contour_coordinates = [self.contour.copy().ravel()]
        if cv2.waitKey() & 0xFF == ord('q'):
            sys.exit(-1)

    def process(self, contour_imgs):
        """ Tracks finger and material, gathering required information. """
        imgs = self._track_objects(contour_imgs)
        self.finger_positions.append(self.finger_position.copy())
        self.contour_coordinates.append(self.contour.copy().ravel())
        return imgs

    def save(self, file_name):
        """ Saves finger positions and coordinates of contour neurons as npz file. """
        np.savez(file_name, X=np.array(self.finger_positions), Y=np.array(self.contour_coordinates))

    def _extract_contours(self, img, num_label=0):
        """ Get external contour from image and show in screen. """
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #print("number of contours = ", len(contours))
        imc = np.zeros(im2.shape)
        if contours != []:
            cnt = max(contours, key=len)
            cv2.drawContours(imc, [cnt], 0, 255, 1)
            cnt = cnt[:,0,:]
        else:
            cnt = np.array([])
        #print(cnt, type(cnt), cnt.shape)
        return cnt, imc

    def _init_material(self, material_contour_img, file_initial_contour, train):
        """ Creates and returns initial contour approximation with GNG. """
        contour, imc = self._extract_contours(material_contour_img, self.material_num_label)
        gng_contour = None
        if train:
            gng = GNG.calibrate_tracking_GNG(contour, tracking_params)
            gng.draw(imc)
            gng_contour = gng.contour()
            np.savetxt(file_initial_contour, gng_contour)
        else:
            gng_contour = np.loadtxt(file_initial_contour)
        #print(gng_contour)
        cv2.imshow('Contour ' + str(self.material_num_label), imc)
        cv2.moveWindow('Contour ' + str(self.material_num_label), 3 * imc.shape[1], self.material_num_label * (imc.shape[0] + Y_SHIFT))
        return gng_contour

    def _draw_ng(self, img):
        """ Draws adapter contour """
        num_coords = len(self.contour)
        contour = self.contour.astype(int)
        for i, coord in enumerate(contour):
            cv2.circle(img, tuple(coord), 3, 255, 1)
            cv2.line(img, tuple(coord), tuple(contour[(i+1)%num_coords]), 255, 2)
        cv2.imshow('Contour ' + str(self.material_num_label), img)

    def _track_material(self, img):
        """ Use NG to track deformable material. """
        print("Tracking material...")
        pixel_contour, imc = self._extract_contours(img, self.material_num_label)
        NG.adapt_NG(self.contour, pixel_contour, **neural_gas_params)
        self._draw_ng(imc)
        return imc

    def _track_finger(self, img):
        """ Detects row, column position of finger. """
        pixel_contour, imc = self._extract_contours(img, self.finger_num_label)
        if pixel_contour != np.array([]):
            coords = np.mean(pixel_contour, 0)
        else:
            coords = np.array([-1, -1], np.float32)
        self.finger_position = coords
        #print(coords)
        #try:
        #    cv2.circle(imc, tuple(coords.astype(int)), 3, 255, 1)
        #except TypeError:
        #    print("Finger is gone")
        #    coords = np.array([-1,-1], np.float32)
        cv2.imshow('Contour ' + str(self.finger_num_label), imc)
        return imc

    def _track_objects(self, contour_imgs):
        """ Function specific to the problem of tracking the finger and material
        pushed by the robot. """
        imm = self._track_material(contour_imgs[self.material_num_label])
        imf = self._track_finger(contour_imgs[self.finger_num_label])
        return imm, imf


def track(cap, param_suit, TrackerClass):
    """ Use NG to track contours"""
    grays = GNG.cluster_colours(num_objects + 1)[param_suit['color_indices']]
    loss_threshold = param_suit['loss_threshold']
    save_file = param_suit['file_save']

    tracker = None
    num_frame = 1

    ## Create GNG from contour in first frame
    if cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Couldn't read first movie frame.", file=sys.stderr)
            return -1
        start_time = time.time()

        images = get_objects(frame, grays, loss_threshold)
        dsts = []
        for i, img in enumerate(images):
            dsts.append(detect_borders(img, param_suit, i))

        tracker = TrackerClass(dsts, param_suit['file_initial_contour'], param_suit['train'])

        print("--- %s seconds to process frame %d ---" % ((time.time() - start_time), num_frame))

        print("Frame ", num_frame)
        num_frame += 1
        #if cv2.waitKey(10) & 0xFF == ord('q'):
        #    return

    else:
        print("Couldn't open movie.", file=sys.stderr)
        return -1

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

        imgs = tracker.process(dsts)

        print("--- %s seconds to process frame %d ---" % ((time.time() - start_time), num_frame))


        print("Frame ", num_frame)
        num_frame += 1
        #if cv2.waitKey(10) & 0xFF == ord('q'):
        #    break
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            img = cv2.addWeighted(imgs[0], 1, imgs[1], 1, 0.0)
            cv2.imwrite(param_suit['file_video'] + '.jpg', img)

    tracker.save(save_file)


def print_usage(script_name):
    """Print script instructions."""
    print("Use: " + sys.argv[0] + " <set_of_paramteres>\nOptions:")
    for key in param_suits.keys():
        print('\t', key)
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

    track(cap, param_suit, FingerMaterialTracker)

    print("Processing finished.  Press key to end program.")
    cv2.waitKey()
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
