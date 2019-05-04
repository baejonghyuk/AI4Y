import numpy as np
import cv2
import sys
from imutils.perspective import four_point_transform
import imutils
import time

# ------------------------------------------------------------------------------------------------------
# Function: To detect Red Marker on the play field.  Red Marker indicate the time to do AI inference
# Inputs:   crop_img - the lower part of the video image crop for Red Marker Detection
#           frame - pass in image for function to draw Red boxes as marker
#           y1 - the amount of y pixel cropped
# return:   ret - 0 = no Red Marker detected, 1 = Red Marker Detected
#           frame - the drawn frame to display out
# ------------------------------------------------------------------------------------------------------
def RedMarkerDetection(crop_img, frame, y1):

    ret = 0
    
    # ------------------------------------------------------------
    # Start: Red Marker detection
    # ------------------------------------------------------------
    frameArea = frame.shape[0]*frame.shape[1]

    # define range HSV for Red color of the Red Marker
    lower_red = np.array([169,100,100])
    upper_red = np.array([189,255,255])

    # convert color image to HSV color scheme
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # define kernel for smoothing   
    kernel = np.ones((3,3),np.uint8)
    # extract binary image with active blue regions
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # define variables to hold values during loop
    largestArea = 0
    largestRect = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        for cnt in cnts:
            # Rotated Rectangle. Here, bounding rectangle is drawn with minimum area,
            # so it considers the rotation also. The function used is cv2.minAreaRect().
            # It returns a Box2D structure which contains following detals -
            # ( center (x,y), (width, height), angle of rotation ).
            # But to draw this rectangle, we need 4 corners of the rectangle.
            # It is obtained by the function cv2.boxPoints()
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
                
            # count euclidian distance for each side of the rectangle
            sideOne = np.linalg.norm(box[0]-box[1])
            sideTwo = np.linalg.norm(box[0]-box[3])
            # count area of the rectangle
            area = sideOne*sideTwo
            # find the largest rectangle within all contours
            if area > largestArea:
                largestArea = area
                largestRect = box
            

    # draw contour of the found rectangle on  the original image
    if largestArea > frameArea*0.02:
        for i in range(0, 4):
            largestRect[i] += [0, y1]
        cv2.drawContours(frame,[largestRect],0,(0,0,255),2)
        ret = 1
        
        
    return ret, frame          
    # ------------------------------------------------------------
    # End: Red Marker detection
    # ------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------
# Function: Main Program
# Inputs:   
# return:   
# ------------------------------------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------------------------------------------
    # Main Program
    # ------------------------------------------------------------------------------------------------------
    video_capture = cv2.VideoCapture('http://admin:@192.168.10.1/media/?action=stream')
    
    while(True):
        # Capture the frames
        grabbed, frame = video_capture.read()
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT )
        
        # Crop the image lower image frame for Line Tracing and detect Red Marker
        y1 = int(height - int(height/4))
        y2 = int(height)
        x1 = 0
        x2 = int(width)
        crop_img = frame[y1:y2, x1:x2]
            
        ret, frame = RedMarkerDetection(crop_img, frame, y1) # Red Marker Detection 
        
        #Display the resulting frame
        cv2.imshow('frame',frame)

        # Handle user keyboard inputs
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
