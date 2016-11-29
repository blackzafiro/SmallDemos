#!/usr/bin/python3.5

import sys
import os
OPENCV_HOME = os.environ['OPENCV_HOME']
sys.path.append(OPENCV_HOME + '/lib/python3.5/dist-packages')

import cv2


def showChannels(img, color_space, ypos = 0, wait=False):
    """ Shows a window with the values in every channel of img. """
    for i in range(img.shape[2]):
        label = color_space[i]
        cv2.imshow(label, img[:,:,i])
        cv2.moveWindow(label, i * img.shape[1], ypos)
    if wait:
        if cv2.waitKey() & 0xFF == ord('q'):
            sys.exit(0)


def change_to(img, color_space):
    if color_space == 'bgr':
        return
    if color_space == 'yuv':
        img[:,:] = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return
    if color_space == 'hsv':
        img[:,:] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return
    if color_space == 'hls':
        img[:,:] = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        return
    if color_space == 'luv':
        img[:,:] = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
        return
    if color_space == 'xyz':
        img[:,:] = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
        return
    if color_space == 'lab':
        img[:,:] = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return


if __name__ == '__main__':
    nargs = len(sys.argv)
    if (nargs < 2):
        print(sys.argv[0] + " <video> <space> [y pos in screen]")
        sys.exit(1)

    cap = cv2.VideoCapture(sys.argv[1])
    color_space = sys.argv[2]
    if nargs > 2:
        ypos = int(sys.argv[3])
    else:
        ypos = 0

    ## Capture first frame
    ret, frame = None, None
    if cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture first frame")
            sys.exit(2)
    else:
        print("Failed to capture source")
        sys.exit(1)

    # Resize
    small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    change_to(small_frame, color_space)
    showChannels(small_frame, color_space, ypos)

    ## End

    if cv2.waitKey() & 0xFF == ord('q'):
        sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()