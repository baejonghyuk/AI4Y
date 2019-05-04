import numpy as np
import cv2
import sys
import serial
from imutils.perspective import four_point_transform
#from imutils import contours
import imutils
import time

# ------------------------------------------------------------------------------------------------------
# Function: to send command (string with new line) to micro:bit
# Input: Cmd - the string command to send (one character)
# Return: none
# ------------------------------------------------------------------------------------------------------
def SerialSendCommand(ser, Cmd):
    Cmd_Str = Cmd + '\n'
    cmd_bytes = str.encode(Cmd_Str)
    ser.write(cmd_bytes)
    
# ------------------------------------------------------------------------------------------------------
# Function: to receive response from micro:bit
# Input: Cmd - the string expecting to receive (one character)
# Return: ret = 1 correct response received, -1 = incorrect/no response received
# ------------------------------------------------------------------------------------------------------
def SerialReceiveResponse(ser, Cmd):
    line = ser.readline()
    text = str(line)        # Convert bytes array to string
    if Cmd in text:
        ret = 1
    else:
        ret = -1
        
    return ret

# ------------------------------------------------------------------------------------------------------
# Function: To execute line tracing
# Inputs:   Ser_Cmd_Str - the serial command sent to micro:bit
#           pre_status - previous command sent to micro:bit
#           cur_status - current command sent to micro:bit
#           tic - timeout ref count
#           crop_img - the lower part of the video image crop for Red Marker Detection
#           width - the width of the video image
#           y1 - the amount of y pixel cropped
# return:   Ser_Cmd_Str
#           prev_status
#           cur_status
#           tic
# ------------------------------------------------------------------------------------------------------
def LineTracing(frame, ser, Ser_Cmd_Str, prev_status, cur_status, tic, crop_img, width, y1):
    # ------------------------------------------------------------
    # Start: Black Line Tracing
    # ------------------------------------------------------------
    # Convert to grayscale
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Color thresholding
    ret,thresh = cv2.threshold(blur,60,255,cv2.THRESH_BINARY_INV)

    # Find the contours of the frame
    _, contours, hierarchy = cv2.findContours(thresh.copy(), 1, cv2.CHAIN_APPROX_NONE)

    # Find the biggest contour (if detected)
    # Left end                                              mid                                             Right end
    # |             cx_lower        cx_lower2               |                  cx_upper2         cx_upper             
    # | <------30%----->|------40%------>|                  |                      |<------40%------|------30%----->|
    # |                                                     |                                                       |
    # | Big Turn L      | Small Turn l   |              Forward                    | Small Turn r   | Big Turn R    |
    cx_upper = int(width - (width * 0.3))
    cx_lower = int(width * 0.3)
    cx_upper2 = int(width - (width * 0.4))
    cx_lower2 = int(width * 0.4)
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)

        if M['m00'] == 0:
            print('divide by zero')
            Ser_Cmd_Str = 's'
            SerialSendCommand(ser, Ser_Cmd_Str)
            Ser_Cmd_Str = ''
            prev_status = -1
        else:
            # not divide by zero
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            cv2.line(frame,(cx,0),(cx,720),(255,0,0),1)
            cv2.line(frame,(0,cy + y1),(1280,cy + y1),(255,0,0),1) 
            cv2.drawContours(frame, contours, -1, (0,255,0), 1, offset = (0,y1))

            if Ser_Cmd_Str == '':
                if cx >= cx_upper:
                    cur_status = 1
                    # if cur_status != prev_status:
                    Ser_Cmd_Str = 'R'
                    SerialSendCommand(ser, Ser_Cmd_Str)
                    print('Turn R')
                    tic = time.time()
                elif cx >= cx_upper2:
                    cur_status = 2
                    # if cur_status != prev_status:
                    Ser_Cmd_Str = 'r'
                    SerialSendCommand(ser, Ser_Cmd_Str)
                    print('Turn r')
                    tic = time.time()      
                elif cx < cx_upper2 and cx > cx_lower2:
                    cur_status = 3
                    if cur_status != prev_status:
                        Ser_Cmd_Str = 'f'
                        SerialSendCommand(ser, Ser_Cmd_Str)
                        print('On Track!')
                        tic = time.time()
                elif cx <= cx_lower:
                    cur_status = 4
                    # if cur_status != prev_status:
                    Ser_Cmd_Str = 'L'
                    SerialSendCommand(ser, Ser_Cmd_Str)
                    print('Turn L')
                    tic = time.time()
                elif cx <= cx_lower2:
                    cur_status = 5
                    # if cur_status != prev_status:
                    Ser_Cmd_Str = 'l'
                    SerialSendCommand(ser, Ser_Cmd_Str)
                    print('Turn l')
                    tic = time.time()
                        
                prev_status = cur_status
            else:
                ret = SerialReceiveResponse(ser, Ser_Cmd_Str)
                if ret == 1:
                    Ser_Cmd_Str = ''
                    print('Respond')

                if (time.time() - tic) > 1.25: # timeout one second
                    Ser_Cmd_Str = ''
                    prev_status = -1
                    print('Timeout')
            # end if Ser_Cmd_Str == '':
        # end if M['m00'] == 0:

    else:
        Ser_Cmd_Str = 's'
        SerialSendCommand(ser, Ser_Cmd_Str)
        Ser_Cmd_Str = ''
        prev_status = -1
        print('I dont see the line')
    # end if len(contours) > 0:
    
    return Ser_Cmd_Str, prev_status, cur_status, tic
    # ------------------------------------------------------------
    # End: Black Line Tracing
    # ------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------
# Function: To send Command and handle response
# Inputs:   Ser_Cmd_Str - the serial command string to send
#           Cmd - the command to send and received
#           tic - current time
#           timeout - time out threshold
# return:   ret - -1 = nothing send, -2 = timeout, 1 = command sent, 2 = received reply
#           Ser_Cmd_Str
#           tic  
# ------------------------------------------------------------------------------------------------------
def SerialCommandNResponse(ser, Ser_Cmd_Str, Cmd, tic, timeout):
    ret = -1
    if Ser_Cmd_Str == '':        
        Ser_Cmd_Str = Cmd
        SerialSendCommand(ser, Ser_Cmd_Str)
        print('Sent ' + Cmd)
        cur_status = 6
        tic = time.time()
        ret = 1
    else:
        ret2 = SerialReceiveResponse(ser, Ser_Cmd_Str)    # end sure stop command sent and received by micro:bit
        if ret2 == 1:
            Ser_Cmd_Str = ''
            print('Respond')
            ret = 2

        if (time.time() - tic) > timeout: # timeout one second
            Ser_Cmd_Str = ''
            print('Timeout')
            ret = -2

    return ret, Ser_Cmd_Str, tic
    
# ------------------------------------------------------------------------------------------------------
# Function: Main Program
# Inputs:   
# return:   
# ------------------------------------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------------------------------------------
    # Global Variable intialization
    # ------------------------------------------------------------------------------------------------------
    cur_status = 0          # current command send to micro:bit
    prev_status = -1        # previous command sent to micro:bit

    Ser_Cmd_Str = ''        # the Serial command string sent to micro:bit
    tic = time.time()       # timeout reference

    # Opening serial port COM25, baud 115200, no parity, no flow control
    ser = serial.Serial('COM35', 115200, timeout=0, parity=serial.PARITY_NONE, rtscts=0)
    
    # ------------------------------------------------------------------------------------------------------
    # Main Program
    # ------------------------------------------------------------------------------------------------------
    video_capture = cv2.VideoCapture('http://admin:@192.168.10.1/media/?action=stream')
    
    while(True):
        # Capture the frames
        grabbed, frame = video_capture.read()
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT )

        # Line tracing and Red Marker Detection 
        # Crop the image lower image frame for Line Tracing and detect Red Marker
        y1 = int(height - int(height/4))
        y2 = int(height)
        x1 = 0
        x2 = int(width)
        crop_img = frame[y1:y2, x1:x2]          
            
        Ser_Cmd_Str, prev_status, cur_status, tic = LineTracing(frame, ser, Ser_Cmd_Str, prev_status, cur_status, tic, crop_img, width, y1)
            
        #Display the resulting frame
        cv2.imshow('frame',frame)

        # Handle user keyboard inputs
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    SerialSendCommand(ser, 's')
    ser.close()
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
