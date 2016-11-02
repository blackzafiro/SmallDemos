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


segment_params = {'max_age' : 40,
                  'lambda_steps' : 1000,        # insert node evey lambda steps
                  'epsilon_beta' : 0.05,        # 0 < beta < 1
                  'epsilon_eta' : 0.0006,       # 0 < eta < 1
                  'alfa' : 0.5,                 # 0 < alfa < 1
                  'beta' : 0.0005               # 0 < beta < 1
                  }

vision_params = {'median_kernel_size' : 7
                }

sponge_set_1 = {'file_video' : '/home/blackzafiro/Programacion/MassSpringIV/data/press/sponge_centre_100.mp4',
                'roi' : ((325,200),(925,700)),
                'file_segment_gng' : 'luv_segment_gng_scentre.pickle',}


## Could try
## http://docs.opencv.org/3.1.0/df/d9d/tutorial_py_colorspaces.html
## as well

def extract_foreground(gng, get_hsv=False):
    """ Separate foreground and background using k-means and value.
    If get_hsv is True it uses Cretu's approach: take mean hsv of foreground and background,
    the one with biggest value in position zero,
    else it returns the k-means centroids with hsv and positon coordinates, the centroid
    with biggest value in position zero.
    """
    observations = np.array(list(gng.nodes.keys()))
    centroids, labels, inertia = cluster.k_means(observations, 2)
    print("Centroides:\n", centroids)

    # The centroids given by k-means correspond to the mean of all elements in the cluster
    # for each dimension
    #print("Etiquetas:\n", labels)

    # Put centroid with greatest value in positon zero (foreground candidate)
    if centroids[0][2] < centroids[1][2]:
        temp = np.copy(centroids[0])
        centroids[0] = centroids[1]
        centroids[1] = temp

    if get_hsv:
        # Use only hsv coordinates and ignore position
        centroids = centroids[:,0:3]
        #print(centroids)
        for node, n_data in gng.nodes.items():
            dist0 = np.linalg.norm(node[0:3] - centroids[0])
            dist1 = np.linalg.norm(node[0:3] - centroids[1])
            if dist0 < dist1:
                n_data.in_foreground = True
            else:
                n_data.in_foreground = False
            #print(node, n_data.in_foreground)
    else:
        for key, label in zip(observations, labels):
            n_data = gng.nodes[tuple(key)]
            if label == 1:
                n_data.in_foreground = True
            else:
                n_data.in_foreground = False
            #print(key, n_data.in_foreground)
    gng.plotNetColorNodes(with_edges=True)
    gng.ax.scatter(centroids[0][0], centroids[0][1], centroids[0][2], c='r', marker='d', s=100)
    gng.ax.scatter(centroids[1][0], centroids[1][1], centroids[1][2], c='b', marker='s', s=100)
    gng.fig.canvas.draw()
    return centroids


def extract_material_finger_background(gng, num_clusters=3):
    """ Separate foreground and background using k-means and value.
    It returns the k-means centroids with hsv, the centroid
    with biggest value in position zero.
    """
    observations = np.array(list(gng.nodes.keys()))[:,0:3]
    kmeans = cluster.KMeans(n_clusters=num_clusters).fit(observations)
    print("Centroids:\n", kmeans.cluster_centers_)

    # Put centroid with greatest value in positon zero (foreground candidate)
    indexes_sorted = kmeans.cluster_centers_[:,2].argsort()
    kmeans.cluster_centers_ = kmeans.cluster_centers_[indexes_sorted]
    kmeans.labels_ = kmeans.labels_[indexes_sorted]

    # TODO:
    # Use only hsv coordinates and ignore position
    centroids = kmeans.cluster_centers_
    print("Sorted by value:\n", centroids)
    dists = np.zeros(len(centroids))
    for node, n_data in gng.nodes.items():
        for i, centroid in enumerate(centroids):
            dists[i] = np.linalg.norm(node[0:3] - centroid)
        clase = np.argmax(dists)
        if clase == 0:
            n_data.in_foreground = True

        else:
            n_data.in_foreground = False
        gng.num_clases = num_clusters
        n_data.clase = clase

    gng.plotNetColorNodes(with_edges=True)
    colors = cm.rainbow(np.linspace(0, 1, len(centroids)))
    for centroid, color in zip(centroids, colors):
        gng.ax.scatter(centroid[0], centroid[1], centroid[2], c=colors, marker='d', s=100)
    gng.fig.canvas.draw()
    return kmeans


def material_finger_background_from_hsv(src, dst, kmeans_class):
    """
    Segments the image in as many clusters as kmeans_class has and
    paints every cluster in a different shade of gray.
    :param src: hsv image
    :param dst: gray scale segmented image
    :param kmeans_class: KMeans instance for classification
    """
    num_classes = len(kmeans_class.cluster_centers_)
    step = math.ceil(255 / (num_classes - 1))
    vals = [min(255, step * i) for i in range(0,4)]

    rows, cols = src.shape[0], src.shape[1]
    for i in range(rows):
        for j in range(cols):
            dst[i,j] = vals[kmeans_class.predict((src[i,j],))[0]]


def background_foreground_from_hsv(src, dst, hsv_centroids):
    """ Uses the hsv coordinates of the centroids to mark each pixel in dst as
    foreground or background
    """
# def background_foreground_from_hsv(src, dst, hsv_centroids, tolerance=(int(0.09*180), int(0.09*255), int(0.02*255))):
# Using the tolerance failed to associate the center of the sponge with the foreground
    rows, cols = src.shape[0], src.shape[1]
    for i in range(rows):
        for j in range(cols):
            # Compeating between clusters:
            dist0 = np.linalg.norm(src[i,j] - hsv_centroids[0])
            dist1 = np.linalg.norm(src[i,j] - hsv_centroids[1])
            dst[i,j] = 255 if dist0 < dist1 else 0

            # Using tolerance
            #print(src[i,j])
            #dif = tolerance - np.abs(src[i,j] - hsv_centroids[0])
            #dst[i, j] = 255 if len(dif[dif < 0]) else 0

def background_foreground_from_kmeans(src, dst, kcentroids):
    rows, cols = src.shape[0], src.shape[1]
    for i in range(rows):
        for j in range(cols):
            node = np.hstack(([i,j], src[i, j]))
            dist0 = np.linalg.norm(node - kcentroids[0])
            dist1 = np.linalg.norm(node - kcentroids[1])
            dst[i, j] = 255 if dist0 < dist1 else 0
            #print(node, dist0, dist1)


### Using only one color channel

class GNGSegmenter:

    def __init__(self, file_segment_gng=None):
        # Get GNG for segmentation
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
        gng.show()
        gng.plotNetColorNodes(with_edges=True)
        self.gng = gng

    def extract_material_finger_background_from_channel(self, num_clusters=3, channel_index=1):
        """ Separate num_clusters using k-means and value.
        It returns the k-means centroids with the color scheme values,
        it sorts the centroids according to the value in channel_index, with the highest located position zero.
        :param num_clusters: Number of classes to identify (each class should be a target object or background.
        :param channel_index: Index of channel that will be used for classification
        :return: None, it just calculates and assigns trained KMeans class instance
        """
        gng = self.gng
        observations = np.array(list(gng.nodes.keys()))[:,[channel_index]]
        #print(observations)
        kmeans = cluster.KMeans(n_clusters=num_clusters).fit(observations)
        #print("Centroids:\n", kmeans.cluster_centers_, '\nLabels:\n', kmeans.labels_)

        # Put centroid with greatest value in positon zero (foreground candidate)
        indexes_sorted = kmeans.cluster_centers_[:,0].argsort()
        kmeans.cluster_centers_ = kmeans.cluster_centers_[indexes_sorted]
        kmeans.labels_ = kmeans.predict(observations)

        #
        centroids = kmeans.cluster_centers_
        #print("Sorted by channel " + str(channel_index) + ":\n", centroids, '\n', kmeans.labels_)
        for node, n_data in gng.nodes.items():
            label = kmeans.predict(((node[channel_index],),))[0]

            # Will keep this only for when I am interested in two classes: foreground and background
            if label == 0:
                n_data.in_foreground = True
            else:
                n_data.in_foreground = False
            gng.num_clusters = num_clusters
            n_data.label = label

        gng.plotNetColorNodes(with_edges=True)
        colors = cm.rainbow(np.linspace(0, 1, len(centroids)))
        for centroid in centroids:
            coords = np.zeros(3)
            coords[channel_index] = centroid
            label = kmeans.predict((centroid,))[0]
            gng.ax.scatter(coords[0], coords[1], coords[2], c=colors[label], marker='d', s=100)
        gng.fig.canvas.draw()
        self.channel_index = channel_index
        self.kmeans = kmeans

    def material_finger_background_from_channel(self, src, dst):
        """
        Segments the image in as many clusters as kmeans_class has and
        paints every cluster in a different shade of gray.
        :param src: hsv image
        :param dst: gray scale segmented image
        """
        kmeans_class = self.kmeans
        channel_index = self.channel_index
        num_classes = len(kmeans_class.cluster_centers_)
        step = math.ceil(255 / (num_classes - 1))
        vals = [min(255, math.ceil(step * i)) for i in range(0, src.shape[2] + 1)]

        rows, cols = src.shape[0], src.shape[1]
        for i in range(rows):
            for j in range(cols):
                dst[i,j] = vals[kmeans_class.predict( ((src[i,j,channel_index],),) )[0]]

    def detect_borders(self, img):
        # Apply median filtering
        dst = cv2.medianBlur(img, vision_params['median_kernel_size'])
        cv2.imshow('Median filtered', dst)
        cv2.moveWindow('Median filtered', dst.shape[1], 400)

        # Use sobel edge detection
        sobelx = np.absolute(cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=1))
        sobely = np.absolute(cv2.Sobel(dst, cv2.CV_64F, 0, 1, ksize=1))
        sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        cv2.imshow('Sobel', sobel)
        cv2.moveWindow('Sobel', 2 * dst.shape[1], 400)

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
    showChannels(small_frame, ypos = 200, wait=True)

    luv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2Luv)
    showChannels(luv, ypos=400, wait=True)
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

    segmenter = GNGSegmenter(param_suit['file_segment_gng'] if param_suit else None)

    # Use K-Means to separate foreground from background
    dst = np.zeros((luv.shape[0], luv.shape[1]), dtype=np.uint8)

    # It doesn't work if we use the positional information as well :P
    #kcentroids = extract_foreground(gng, False)
    #background_foreground_from_kmeans(hsv, dst, kcentroids)

    #hsv_centroids = extract_material_finger_background(gng)
    #hsv_classes = extract_material_finger_background(gng)
    segmenter.extract_material_finger_background_from_channel()
    #if cv2.waitKey() & 0xFF == ord('q'):
    #    sys.exit(0)
    #material_finger_background_from_hsv(hsv, dst, hsv_classes)
    segmenter.material_finger_background_from_channel(luv, dst)
    #background_foreground_from_hsv(hsv, dst, hsv_centroids)

    cv2.imshow('Segmented', dst)
    cv2.moveWindow('Segmented', 0, 400)

    # Apply median filtering
    # dst = cv2.medianBlur(dst, vision_params['median_kernel_size'])
    # cv2.imshow('Median filtered', dst)
    # cv2.moveWindow('Median filtered', dst.shape[1], 400)
    #
    # # Use sobel edge detection
    # sobelx = np.absolute(cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=1))
    # sobely = np.absolute(cv2.Sobel(dst, cv2.CV_64F, 0, 1, ksize=1))
    # sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    # cv2.imshow('Sobel', sobel)
    # cv2.moveWindow('Sobel', 2 * dst.shape[1], 400)
    dst = segmenter.detect_borders(dst)

    if cv2.waitKey() & 0xFF == ord('q'):
        sys.exit(0)

    cv2.destroyAllWindows()
    print("Start tracking...")

    while(cap.isOpened()):
        ## Capture frame-by-frame
        ret, frame = cap.read()
        if not ret: break

        ## Our operations on the frame come here
        small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        luv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2Luv)
        dst = segmenter.detect_borders(luv)

        ## Display the resulting frame
        cv2.imshow('luv', dst)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
