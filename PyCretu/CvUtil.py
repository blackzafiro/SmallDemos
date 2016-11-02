import cv2
import sys

def showChannels(img, ypos = 0, wait=False):
    """ Shows a window with the values in every channel of img. """
    for i in range(img.shape[2]):
        label = 'Channel ' + str(i)
        cv2.imshow(label, img[:,:,i])
        cv2.moveWindow(label, i * img.shape[1], ypos)
    if wait:
        if cv2.waitKey() & 0xFF == ord('q'):
            sys.exit(0)