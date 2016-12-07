import cv2
import sys

def showChannels(img, ypos = 0, wait=False):
    """ Shows a window with the values in every channel of img. """
    num_channels = img.shape[2] if len(img.shape) == 3 else 1
    if num_channels == 1:
        label = 'One channel'
        cv2.imshow(label, img)
        cv2.moveWindow(label, 0, ypos)
    else:
        for i in range(num_channels):
            label = 'Channel ' + str(i)
            cv2.imshow(label, img[:,:,i])
            cv2.moveWindow(label, i * img.shape[1], ypos)
    if wait:
        if cv2.waitKey() & 0xFF == ord('q'):
            sys.exit(0)

def conversion_to_string(cv_code):
    """
    Returns a string representation of the color conversion
    :param cv_code:
    :return:
    """
    strs = {cv2.COLOR_BGR2GRAY:"bgr to gray",
            cv2.COLOR_BGR2HSV:"bgr to hsv",
            cv2.COLOR_BGR2Luv:"bgr to Luv",
            cv2.COLOR_BGR2XYZ:"bgr to xyz",
            cv2.COLOR_BGR2YCrCb:"bgr to YCrCb",
            cv2.COLOR_BGR2HLS:"bgr to hls",
            cv2.COLOR_BGR2Lab: "bgr to Lab"}
    return strs[cv_code]