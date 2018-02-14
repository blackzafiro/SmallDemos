#!/usr/bin/python3.5
import os, sys
OPENCV_HOME = os.environ['OPENCV_HOME']
sys.path.append(OPENCV_HOME + '/lib/python3.5/dist-packages')
sys.path.append("./")

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from FFPredict import load_data

param_suit = {
    'file_train_data': 'data/pickles/sponge_set_1_track.npz',
    'file_force_data': 'data/original/sponge_centre_100.txt',
    'file_video': 'data/original/sponge_centre_100.mp4'
}


def plot_history(Y):
    """ Plots the trajectories of the neurons through time. """
    fig = plt.figure("Control points' history")
    fig.suptitle("Control points' history")
    ax = fig.add_subplot(111, projection='3d')

    NNEURONS = int(Y.shape[1]/2)
    values = range(NNEURONS)
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    t = np.arange(Y.shape[0])
    for i in range(0, Y.shape[1], 2):
        x = Y[:,i]
        y = -1 * Y[:,i+1]

        colorVal = scalarMap.to_rgba(values[int(i/2)])
        ax.plot(t, x, y, color=colorVal)

    ax.set_xlabel('time (steps)')
    ax.set_ylabel('x (pixels)')
    ax.set_zlabel('y (pixels)')
    plt.show()

if __name__ == '__main__':
    X, Y = load_data(param_suit['file_train_data'],
                     param_suit['file_force_data'])
    plot_history(Y)
