import msvcrt
import serial
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
    Ser_Cmd_Str = ''        # the Serial command string sent to micro:bit
    tic = time.time()       # timeout reference

    # Opening serial port COM25, baud 115200, no parity, no flow control
    ser = serial.Serial('COM4', 115200, timeout=0, parity=serial.PARITY_NONE, rtscts=0)

    Cmd2Send = ''

    print('Input your commands: f, s, l, r, L, R')
    # ------------------------------------------------------------------------------------------------------
    # Main Program
    # ------------------------------------------------------------------------------------------------------
    while(True):

        if Cmd2Send == '':
            if msvcrt.kbhit():
                key = msvcrt.getch()
                # print('You hit ' + str(key))
                # Handle user keyboard inputs
                if key == b'q':
                    break
                elif key == b'f':
                    Cmd2Send = 'f'
                    print('Forward')
                elif key == b'l':
                    Cmd2Send = 'l'
                    print('Small Left Turn')
                elif key == b'r':
                    Cmd2Send = 'r'
                    print('Small Right Turn')
                elif key == b's':
                    Cmd2Send = 's'
                    print('Stop')
                elif key == b'R':
                    Cmd2Send = 'R'
                    print('Big Right Turn')
                elif key == b'L':
                    Cmd2Send = 'L'
                    print('Big Left Turn')
        else:
            ret, Ser_Cmd_Str, tic = SerialCommandNResponse(ser, Ser_Cmd_Str, Cmd2Send, tic, 2)
            if ret == 2:
                Cmd2Send = ''
                print('Command Done!')



    SerialSendCommand(ser, 's')
    ser.close()

if __name__ == '__main__':
    main()
