import numpy as np
import cv2
import sys
import serial
from imutils.perspective import four_point_transform
#from imutils import contours
import imutils
from keras.models import model_from_json
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
# Function: To detect Traffic Sign
# Inputs:
# return:
# ------------------------------------------------------------------------------------------------------
def findTrafficSign(grabbed, frame):
    '''
    This function find blobs with blue color on the image.
    After blobs were found it detects the largest square blob, that must be the sign.
    '''
    # define range HSV for blue color of the traffic sign
    lower_blue = np.array([85,100,70])
    upper_blue = np.array([115,255,255])

    if not grabbed:
        print("No input image")
        return

    # frame = imutils.resize(frame, width=500)
    frameArea = frame.shape[0]*frame.shape[1]

    # convert color image to HSV color scheme
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define kernel for smoothing
    kernel = np.ones((3,3),np.uint8)
    # extract binary image with active blue regions
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # defite string variable to hold detected sign description
    detectedTrafficSign = None

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
    if largestArea > frameArea*0.02:    # 0.02
        # for i in range(0, 4):
        #    largestRect[i] += [0, 100]
        cv2.drawContours(frame,[largestRect],0,(0,0,255),2)

    cropped = None
    if largestRect is not None:
        # cropped interesting area
        cropped = four_point_transform(frame, [largestRect][0])

        # show an image if rectangle was found
        # cv2.imshow("Warped", cv2.bitwise_not(warped))

        # use function to detect the sign on the found rectangle
        # detectedTrafficSign = identifyTrafficSign(warped)
        # print(detectedTrafficSign)

        # write the description of the sign on the original image
        # cv2.putText(frame, detectedTrafficSign, tuple(largestRect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # show original image
        # cv2.imshow("crop", cropped)

    return frame, cropped, largestRect

# ------------------------------------------------------------------------------------------------------
# Function: To read Traffic Sign
# Inputs:
# return:
# ------------------------------------------------------------------------------------------------------
def ReadTrafficSign(orig_image, model):
    ret = -1

    # Check if the stream is over
    if orig_image is None:
        return ret

    # Scale to the dimension entered by the model, adjust the value range from 0 to 1.
    input_width = 48
    input_height = 48
    resized_image = cv2.resize(
        orig_image,
        (input_width, input_height),
    ).astype(np.float32)
    normalized_image = resized_image / 255.0

    # Execution forecast
    batch = normalized_image.reshape(1, input_height, input_width, 3)
    result_onehot = model.predict(batch)
    obj1_score, obj2_score, obj3_score, obj4_score = result_onehot[0]
    class_id = np.argmax(result_onehot, axis=1)[0]

    if class_id == 0:
        class_str = 'left'
        score = obj1_score
        ret = 1
    elif class_id == 1:
        class_str = 'right'
        score = obj2_score
        ret = 1
    elif class_id == 2:
        class_str = 'stop'
        score = obj3_score
        ret = 1
    elif class_id == 3:
        class_str = 'U_Turn'
        score = obj4_score
        ret = 1

    return ret, class_str, score

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
    MODE_LINE_TRACING = 0
    MODE_JUNCTION_STOP = 1
    MODE_INFERENCE = 2
    MODE_JUNCTION_MANEUVER = 3

    mode_status = MODE_LINE_TRACING     # mode to indicate line tracing, inference or junction maneuver

    cur_status = 0          # current command send to micro:bit
    prev_status = -1        # previous command sent to micro:bit

    Ser_Cmd_Str = ''        # the Serial command string sent to micro:bit
    tic = time.time()       # timeout reference

    # Opening serial port COM25, baud 115200, no parity, no flow control
    ser = serial.Serial('COM4', 115200, timeout=0, parity=serial.PARITY_NONE, rtscts=0)

    # ------------------------------------------------------------------------------------------------------
    # Main Program
    # ------------------------------------------------------------------------------------------------------
    video_capture = cv2.VideoCapture('http://admin:@192.168.10.1/media/?action=stream')
    # video_capture.set(3, 160)
    # video_capture.set(4, 120)

    # Loading model
    with open('C:/AI Car/Data/model.json', 'r') as file_model:
        model_desc = file_model.read()
        model = model_from_json(model_desc)

    model.load_weights('C:/AI Car/Data/weights.h5')

    while(True):
        # Capture the frames
        grabbed, frame = video_capture.read()
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT )

        if mode_status == MODE_LINE_TRACING:
            # Line tracing and Red Marker Detection
            # Crop the image lower image frame for Line Tracing and detect Red Marker
            y1 = int(height - int(height/4))
            y2 = int(height)
            x1 = 0
            x2 = int(width)
            crop_img = frame[y1:y2, x1:x2]

            ret, frame = RedMarkerDetection(crop_img, frame, y1) # Red Marker Detection
            if ret == 1:
                # Red Marker detected -> goto inference mode
                mode_status = MODE_JUNCTION_STOP
                prev_status = -1
                Ser_Cmd_Str = ''
            else:
                # No Red Marker detect -> do line tracing
                Ser_Cmd_Str, prev_status, cur_status, tic = LineTracing(frame, ser, Ser_Cmd_Str, prev_status, cur_status, tic, crop_img, width, y1)

        elif mode_status == MODE_JUNCTION_STOP:
            # AI Inference mode
            ret, Ser_Cmd_Str, tic = SerialCommandNResponse(ser, Ser_Cmd_Str, 's', tic, 1.25)
            if ret == 2:
                mode_status = MODE_INFERENCE

        elif mode_status == MODE_INFERENCE:
            frame, cropped, largestRect = findTrafficSign(grabbed, frame)

            if cropped is not None:
                ret, detectedTrafficSign, score = ReadTrafficSign(cropped, model)
                if ret == 1:
                    percent = score *100
                    printTrafficSign = detectedTrafficSign + ' ' + str('%.2f' % percent) + '%'
                    cv2.putText(frame, printTrafficSign, tuple(largestRect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    if score > 0.9:
                        if 'left' in detectedTrafficSign:
                            mode_status = MODE_JUNCTION_MANEUVER
                            junc_str = 'a'
                            Ser_Cmd_Str = ''
                            prev_status = -1
                        elif 'right' in detectedTrafficSign:
                            mode_status = MODE_JUNCTION_MANEUVER
                            junc_str = 'b'
                            Ser_Cmd_Str = ''
                            prev_status = -1
                        elif 'U_Turn' in detectedTrafficSign:
                            mode_status = MODE_JUNCTION_MANEUVER
                            junc_str = 'u'
                            Ser_Cmd_Str = ''
                            prev_status = -1

        elif mode_status == MODE_JUNCTION_MANEUVER:
            ret, Ser_Cmd_Str, tic = SerialCommandNResponse(ser, Ser_Cmd_Str, junc_str, tic, 6)
            if ret == 2:
                Ser_Cmd_Str = ''
                prev_status = -1
                mode_status = MODE_LINE_TRACING
                print('Respond')

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
