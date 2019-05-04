import numpy as np
import cv2
import sys
   
# ------------------------------------------------------------------------------------------------------
# Function: Main Program
# Inputs:   
# return:   
# ------------------------------------------------------------------------------------------------------
def main():
    video_capture = cv2.VideoCapture('http://admin:@192.168.10.1/media/?action=stream')
    
    while(True):
        # Capture the frames
        grabbed, frame = video_capture.read()
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT )

        #Display the resulting frame
        cv2.imshow('frame',frame)

        # Handle user keyboard inputs
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
