#!/usr/bin/python3.5

import sys
import os
OPENCV_HOME = os.environ['OPENCV_HOME']
sys.path.append(OPENCV_HOME + '/lib/python3.5/dist-packages')

from pathlib import Path
import time

import numpy as np
import cv2

import GNG
import pickle

segment_params = {'max_age' : 40,
                  'lambda_steps' : 1000,
                  'epsilon_beta' : 0.05,        # 0 < beta < 1
                  'epsilon_eta' : 0.0006,       # 0 < eta < 1
                  'alfa' : 0.5,                 # 0 < alfa < 1
                  'beta' : 0.0005               # 0 < beta < 1
                  }

sponge_set_1 = {'file_video' : '/home/blackzafiro/Programacion/MassSpringIV/data/press/sponge_centre_100.mp4',
                'roi' : ((325,200),(925,700)),
                'file_segment_gng' : 'segment_gng_scentre.pickle',}

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
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    if param_suit:
        roi_coords = param_suit['roi']
        cv2.rectangle(frame, roi_coords[0], roi_coords[1], 255)
        cv2.imshow('Source', frame)
        if cv2.waitKey() & 0xFF == ord('q'):
            sys.exit(0)
        else:
            cv2.destroyWindow('Source')
    
    # Display hsv frame
    cv2.imshow('hsv', hsv)
    cv2.moveWindow('hsv', 0, 0)
    if cv2.waitKey() & 0xFF == ord('q'):
        sys.exit(0)
        
    # Get GNG for segmentation
    gng = None
    if param_suit:
        seg_path = Path(param_suit['file_segment_gng'])
        if seg_path.is_file():
            with seg_path.open(mode = 'rb') as f:
                gng = pickle.load(f)
                if hasattr(gng, 'fig'):
                    del gng.fig
    if gng is None:
        start_time = time.time()
        gng = GNG.calibrateSegmentationGNG(hsv, segment_params)
        print("--- %s seconds to callibrate segmentation GNG ---" % (time.time() - start_time))
        if param_suit:
            with open(param_suit['file_segment_gng'], 'wb') as f:
                pickle.dump(gng, f)
    hsv_mean = gng.calculate_foreground_mean()
    gng.show()
    gng.plotHSV(with_edges = True)
    
    if cv2.waitKey() & 0xFF == ord('q'):
        sys.exit(0)
      
    #while(cap.isOpened()):
        ## Capture frame-by-frame
        #ret, frame = cap.read()
        #if not ret: break

        ## Our operations on the frame come here
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        ## Display the resulting frame
        #cv2.imshow('hsv', hsv)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
