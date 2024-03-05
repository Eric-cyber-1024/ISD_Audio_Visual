'''
Dependency:
opencv-contrib-python==4.7.0.72

'''

from pywinauto import Desktop  # add this to handle UI scaling issue
import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import yaml
import datetime
import paramiko
import tkinter as tk
from tkinter import simpledialog
import serial
import socket
import selectors
import types
import keyboard

try:
    from roypypack import roypy  # package installation
except ImportError:
    import roypy  # local installation
import queue
import sys
import threading
from sample_camera_info import print_camera_info
from roypy_sample_utils import CameraOpener, add_camera_opener_options
from roypy_platform_utils import PlatformHelper
from data_logger import DataLogger
from test_delay_cal import *



# Add[support for pmd flexx2 depth camera],Brian,29 Feb 2024
K=4
DEPTH_CAM_WIDTH =224
DEPTH_CAM_HEIGHT=172

remoteHostIPAddr=''
endDepthCamFlag=False
applyEqHistDepthCamFlag=False
targetPosDepthCam='' # target position returned from depth camera, in string format

mouseX = 0
mouseY = 0
LogInterval=1  # secs to log data once
prev_save_time = time.time()  # Initialize the previous save time

# Add,Brian,05 Mar 2024
sel = selectors.DefaultSelector()
sendBuf=b'SET0'
RecvBuf=[]
micIndx = 1          # microphone index, from 1 to 29 
nextMicFlag=False    # True to get delays from FPGA from mic at index micIndx
nextCycleFlag=False  # True to indicate the start of a new cycle
sendFlag=False
stopSerialThread=True# True to stop serial port thread
MIC_NUMBER=32
INDEX =[x for x in range (MIC_NUMBER )]


# add,Brian,05 Mar 2024
def prepareMicDelaysPacket(payload):
    '''
    prepare packet for mic delays

    payload -- list of bytes to be sent 

    return packet = payload + checksum (xor all bytes in payload)

    '''
    checksum = 0
    for byte in payload:
        checksum ^= byte
    packet = bytes(payload) + bytes([checksum])

    return packet

# add,Brian,05 Mar 2024
def validateMicDelaysPacket(packet):
    payload = packet[:-1]
    checksum = packet[-1]

    calculated_checksum = 0
    for byte in payload:
        calculated_checksum ^= byte

    if checksum == calculated_checksum:
        return True
    else:
        return False


def start_connections(host, port):
    server_addr = (host, port)
    print("starting connection to ",  server_addr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(server_addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    data = types.SimpleNamespace(
            outb=b"",
    )
    sel.register(sock, events,data=data)

def service_connection(key, mask,host,port):
    sock = key.fileobj
    data = key.data
    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(1024)  # Should be ready to read
        if recv_data:
            print("received", repr(recv_data) )
            print("closing connection")
            sel.unregister(sock)
            sock.close()
        elif not recv_data:
            print("closing connection")
            sel.unregister(sock)
            sock.close()
    if mask & selectors.EVENT_WRITE:
        global sendFlag
        if sendFlag==True:      
            data.outb = sendBuf
            print("sending", repr(data.outb), "to connection", host, port)
            sent = sock.send(data.outb)  # Should be ready to write
            sendFlag=False

def struct_packet(RW_field,mode,mic_gain,mic_num,en_bm,en_bc,delay_binary_output_x,mic_en,type,reserved):
     #convert into binary form 
#     MIC_change = np.asarray(list(map(int,bin (int(MIC_change))[2:].zfill(4))))
#     MIC_change = [0,1,1,1]
     this_mic_no  = decToBin(mic_num,5)
     this_type=decToBin(type,2)
     this_reserved=decToBin(reserved,4)
#     MC_BM_packet = BM_MC_status(BM_MC_bit_received)
#     delay_time   = delay_binary_output
#     mic_ONOFF = mic_onoff_ID
 #    Channel = np.asarray(list(map(int,bin (int(choice))[2:].zfill(2)))) # register change ch num and turn into binary
     #putting them into array order 
     return np.hstack((RW_field,mode,mic_gain,this_mic_no,en_bm,en_bc,delay_binary_output_x,mic_en,this_type,this_reserved))


# create mic ID in Binary form 
def decToBin(this_value,bin_num):
    num = bin(this_value)[2:].zfill(bin_num)
    num = list(map(int,num)) # convert list str to int 
    return num 

def BintoINT(Binary):
    integer = 0 
    result_hex = 0 
    for x in range(len(Binary)):
        integer = integer + Binary[x]*2**(len(Binary)-x-1)
    result_hex = hex(integer)
    return result_hex



# Function to read data from the serial port
def read_serial_data(ser):
    global nextCycleFlag,nextMicFlag, micIndx, stopSerialThread,debug_message,targetPosDepthCam,loggerFPGA,loggerDebug

    while not stopSerialThread:
        line = ser.readline().decode('ISO-8859-1').strip()
        print(line)
        
        
        if line.find('Read after trigger')>=0:
            print(line)
            # log fpga message from UART
            loggerFPGA.add_data(line)
            
            # if its the last shot, print sth to indicate
            if line.find('j=7')>0:
                print('ready for next microphone...., currently at mic# %d' %(micIndx))
                # increment micIndx if micIndx<29
                if micIndx<29:
                    micIndx+=1
                    nextMicFlag=True
                else:
                    # reset micIndex and nextMicFlag
                    micIndx = 1
                    nextMicFlag=True
                    # raise wait next cycle flag
                    nextCycleFlag=True

    print('leaving serial port thread')


def fpgaCommHandler():

    global sendFlag, sendBuf,sel,nextCycleFlag,nextMicFlag,micIndx,debug_message

    RW_field=[1,1]
    mode=0
    mic_gain=[1,0]
    mic_num=0
    en_bm=1
    en_bc=1
    mic_en=1
    type=0
    reserved=0
    dummy = [0,0,0,0,1,0,0,1,1,0,1,0,0]
    message=struct_packet(RW_field,mode,mic_gain,mic_num,en_bm,en_bc,dummy,mic_en,type,reserved)
    #print(message)
    messagehex = BintoINT(message)
    print(messagehex)
    message1 = int(messagehex[2:4],16) # hex at  1 and 2  
    message2 = int(messagehex[4:6],16) # hex at  3 and 4 
    message3 = int(messagehex[6:8],16)  # hex at  5 and 6 
    message4 = int(messagehex[8:],16)
    print("m1:{},m2:{},m3:{},m4:{}\n".format(message1,message2,message3,message4))

    inStr = input('Please input FPGA commandline here...')
    
    while len(inStr.split())!=8:
        inStr = input('incorrect number of arguments, please try again')
    
    host, port, mode, mic, mic_vol, mic_disable, set_test,mic_delay = inStr.split()

    message5 = int(mode)
    message6 = int(mic)
    message7 = int(mic_vol)
    message8 = int(mic_disable)
    message9 = int(set_test)
    message10 = int(mic_delay)

    
    nextCycleFlag=True # start with a new cycle
    nextMicFlag = True # enabled to get mic delays
    micIndx     = 1    # always start from m#1 
    calibratedFlag=False # True to send estimated mic delays to FPGA
    while True:
        try:
            # wait until nextMicFlag is True
            while not nextMicFlag:
                time.sleep(0.1)

                if keyboard.is_pressed('q'):  # Check if 'q' key is pressed
                    print('q pressed')

            if nextCycleFlag:
                inStr = input("press any keys to start a new cycle...")
                nextCycleFlag=False
            
            # sleep for 3 secs first
            time.sleep(3)
            nextMicFlag=False
            sMsg = '****** Selected mic index=%d' %(micIndx)
            print(sMsg)
            loggerFPGA.add_data(sMsg)
            message6=micIndx
            
            sel = selectors.DefaultSelector()     #wx add can work looply
            start_connections(host, int(port))

            sendBuf=bytes([message1,message2,message3,message4,message5,message6,message7,message8,message9,message10])

            # obtain mic delays from 3d coordinate or zeros
            payload = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            if mode=='2':
                # estimate 32 mic delays from 3d coordinate
                
                # step 1, get 3d coordinates from debug_message
                # Parse the string and extract the first three float values
                values = debug_message.split(",")[:3]
                vec = np.array([float(value) for value in values])
                print(vec)
                _,refDelay,_ = delay_calculation(vec)   
                refDelay = refDelay*48e3
                refDelay = np.max(refDelay)-refDelay
                refDelay = np.round(refDelay)

                #convert refDelay to byte
                #but make sure that they are within 0 to 255 first!!
                assert (refDelay>=0).all() and (refDelay<=255).all()

                # payload = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
                #         0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,0x1f,0x20]

                refDelay = refDelay.astype(np.uint8)
                payload = refDelay.tobytes()
                print(refDelay)
                print(payload)
                print(sendBuf)

                

            packet = prepareMicDelaysPacket(payload)
            
            sendBuf+=packet
            print(sendBuf)
            sendFlag=True
            while True:
                events = sel.select(timeout=None)
                if events:
                    for key, mask in events:
                        service_connection(key, mask,host,int(port))
                # Check for a socket being monitored to continue.
                if not sel.get_map():
                    print("exit 2")
                    break
        except KeyboardInterrupt:
            print("caught keyboard interrupt, exiting")
            break
        finally:
            print("exit 3")
            sel.close()


def showDialogSelectTalkboxLocIndex():
    root = tk.Tk()
    root.withdraw()

    number = simpledialog.askinteger("Select Talkbox Loc Index", "Select Talkbox Loc Index")
    
    if number is None:
        return -1
    else:
        return number


# code to run on remote rpi3 
# Function to execute the SSH command
def execute_ssh_command(hostname,sCmd):
    # Create an SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote server
        client.connect(hostname, username='pi', password='pi')

        # Run the remote command
        stdin, stdout, stderr = client.exec_command(sCmd)

        # Read the output of the command
        output = stdout.read().decode()

        # Print the output
        print(output)
        logger.add_data(output)
        

    finally:
        # Close the SSH connection
        client.close()

def set_testrig_XY_byIndex(hostname,index):

    sPortName='/dev/ttyACM0'
    sCmd = 'cd ~/workspace/python ; python test_xy_uart.py %s %d' %(sPortName,index)
    
    # Create a thread for executing the SSH command
    ssh_thread = threading.Thread(target=execute_ssh_command,args=(hostname,sCmd))

    # Start the thread
    ssh_thread.start()

    # Wait for the SSH thread to complete
    ssh_thread.join()


def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y
    


class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.frame = 0
        self.done = False
        self.undistortImage = False
        self.lock = threading.Lock()
        self.once = False
        self.queue = q

    def get3DCoordinates(self,u,v,z):
        global targetPosDepthCam

        M = self.cameraMatrix
        fx= M[0][0]
        fy= M[1][1]
        cx= M[0][2]
        cy= M[1][2]
        x = (u-cx)*(z/fx)
        y = (v-cy)*(z/fy)

        # save x,y,z to targetPosDepthCam
        targetPosDepthCam = 'dcam,%.2f,%.2f,%.2f' %(x,y,z)
        #print(targetPosDepthCam)

    

    def onNewData(self, data):
        p = data.npoints()
        self.queue.put(p)
        

    def paint (self, data):

        global stacked_frame
        """Called in the main thread, with data containing one of the items that was added to the
        queue in onNewData.
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.
        font_color = (0, 0, 255)  # White color
        line_type = cv2.LINE_AA

        # Write the debug message on the image
        
        text_size, _ = cv2.getTextSize("Debug Message", font, font_scale, 1)
        text_x = 10
        text_y = 10 + text_size[1]
        background_color = (50, 50, 50)  # dark grey background color

        # mutex to lock out changes to the distortion while drawing
        self.lock.acquire()

        depth = data[:, :, 2]   # float64
        gray = data[:, :, 3]    # float64
        confidence = data[:, :, 4]

        

        #print(depth[172//2][224//2])
        
        # comment out, Brian, 29 Feb 2024
        yy = (mouseY%(DEPTH_CAM_HEIGHT*K))//K
        xx = mouseX//K
        z = depth[yy][xx]
        self.get3DCoordinates(xx,yy,z)
        
        


        # image size = 172 x 224 
        zImage = np.zeros(depth.shape, np.float32)
        grayImage = np.zeros(depth.shape, np.float32)

        #print(zImage.shape,grayImage.shape)


        # iterate over matrix, set zImage values to z values of data
        # also set grayImage adjusted gray values
        xVal = 0
        yVal = 0
        for x in zImage:        
            for y in x:
                if confidence[xVal][yVal]> 0:
                  zImage[xVal,yVal] = self.adjustZValue(depth[xVal][yVal])
                  grayImage[xVal,yVal] = self.adjustGrayValue(gray[xVal][yVal])
                yVal=yVal+1
            yVal = 0
            xVal = xVal+1

        zImage8 = np.uint8(zImage)
        grayImage8 = np.uint8(grayImage)

        # apply undistortion
        if self.undistortImage:
            zImage8 = cv2.undistort(zImage8,self.cameraMatrix,self.distortionCoefficients)
            grayImage8 = cv2.undistort(grayImage8,self.cameraMatrix,self.distortionCoefficients)

        # equalize grayImage8 if needed
        if applyEqHistDepthCamFlag:
            grayImage8 = cv2.equalizeHist(grayImage8)
        else:
            # Create a CLAHE object
            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            # Apply CLAHE to the image
            #grayImage8 = clahe.apply(grayImage8)
            pass

        # finally show the images
        
        # turn zImage8,grayImage8 to color first
        zImg = cv2.cvtColor(cv2.resize(zImage8,[zImage8.shape[1]*K,zImage8.shape[0]*K]),cv2.COLOR_GRAY2BGR)
        gImg = cv2.cvtColor(cv2.resize(grayImage8,[grayImage8.shape[1]*K,grayImage8.shape[0]*K]),cv2.COLOR_GRAY2BGR)
        
        #print(mouseX,mouseY)
        



        # Vertically stack the frames
        stacked_frame = np.vstack((zImg,gImg))
        # marker mouseX, mouseY on stacked_frame
        cv2.circle(stacked_frame,(mouseX,mouseY%(DEPTH_CAM_HEIGHT*K)),3,(0,0,255),-1)
        cv2.circle(stacked_frame,(mouseX,mouseY%(DEPTH_CAM_HEIGHT*K)+(DEPTH_CAM_HEIGHT*K)),3,(0,0,255),-1)


        # Get the text size
        (text_width, text_height), _ = cv2.getTextSize(targetPosDepthCam, font, font_scale, 1)

        # Calculate the background rectangle coordinates
        background_rect_coords = ((0, 0), (text_width + 10, text_height + 10))  # Adjust padding as needed

        # Draw the background rectangle
        cv2.rectangle(stacked_frame, background_rect_coords[0], background_rect_coords[1], background_color, cv2.FILLED)
        cv2.putText(stacked_frame, targetPosDepthCam, (text_x, text_y), font, font_scale, font_color, 2, line_type)
    

        cv2.imshow('depth_plus_IR',stacked_frame)
        cv2.setMouseCallback('depth_plus_IR',draw_circle)
        
        self.lock.release()
        self.done = True

    def setLensParameters(self, lensParameters):
        # Construct the camera matrix
        # (fx   0    cx)
        # (0    fy   cy)
        # (0    0    1 )
        self.cameraMatrix = np.zeros((3,3),np.float32)
        self.cameraMatrix[0,0] = lensParameters['fx']
        self.cameraMatrix[0,2] = lensParameters['cx']
        self.cameraMatrix[1,1] = lensParameters['fy']
        self.cameraMatrix[1,2] = lensParameters['cy']
        self.cameraMatrix[2,2] = 1

        #print(self.cameraMatrix)

        # Construct the distortion coefficients
        # k1 k2 p1 p2 k3
        self.distortionCoefficients = np.zeros((1,5),np.float32)
        self.distortionCoefficients[0,0] = lensParameters['k1']
        self.distortionCoefficients[0,1] = lensParameters['k2']
        self.distortionCoefficients[0,2] = lensParameters['p1']
        self.distortionCoefficients[0,3] = lensParameters['p2']
        self.distortionCoefficients[0,4] = lensParameters['k3']

        print(self.distortionCoefficients)

    def toggleUndistort(self):
        self.lock.acquire()
        self.undistortImage = not self.undistortImage
        self.lock.release()

    # Map the depth values from the camera to 0..255
    def adjustZValue(self,zValue):
        clampedDist = min(7.9,zValue)
        newZValue = clampedDist / 7.9 * 255
        return newZValue

    # Map the gray values from the camera to 0..255
    def adjustGrayValue(self,grayValue):
        clampedVal = min(600,grayValue)
        newGrayValue = (clampedVal/600.)*255
        return newGrayValue

def process_event_queue (q, painter):
    global endDepthCamFlag
    global applyEqHistDepthCamFlag

    while not endDepthCamFlag:
        try:
            
            # try to retrieve an item from the queue.
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit
            if len(q.queue) == 0:
                item = q.get(True, 1)
            else:
                for i in range (0, len (q.queue)):
                    item = q.get(True, 1)
        except queue.Empty:
            # this will be thrown when the timeout is hit
            break
        else:
            painter.paint(item)
            # waitKey is required to use imshow, we wait for 1 millisecond
            currentKey = cv2.waitKey(1)
            if currentKey == ord('d'):
                painter.toggleUndistort()
            elif currentKey == ord('h'):
                # toggle applying Histogram Equalization
                applyEqHistDepthCamFlag = not applyEqHistDepthCamFlag

            # close if esc/q pressed
            elif currentKey == 27 or currentKey == ord('q'):
                break

# depth camera main
def depthcam_main():

    # Set the available arguments
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser (usage = __doc__)
    add_camera_opener_options (parser)
    parser.add_argument('-com','--com',type=str,required=True,help='comport name')
    parser.add_argument('-cam','--cam',type=str,required=True,help='camera index')
    parser.add_argument("-ip","--ip",type=str,required=True,help="remote rpi3 ip address")
    
    options = parser.parse_args()

    # delete some previous arguments
    delattr(options,"ip")
    delattr(options,"cam")
    delattr(options,'com')
    
    # for testing only
    #options.rrf = 'meetingroom4.rrf'
    opener = CameraOpener (options)

    try:
        cam = opener.open_camera ()
    except:
        print("could not open Camera Interface")
        sys.exit(1)

    try:
        # retrieve the interface that is available for recordings
        replay = cam.asReplay()
        print ("Using a recording")
        print ("Framecount : ", replay.frameCount())
        print ("File version : ", replay.getFileVersion())
    except SystemError:
        print ("Using a live camera")

    q = queue.Queue()
    l = MyListener(q)
    cam.registerDataListener(l)
    cam.startCapture()

    lensP = cam.getLensParameters()
    l.setLensParameters(lensP)

    process_event_queue (q, l)

    cam.stopCapture()
    print("Done")


    


# Add[for getting moving averages],Brian,29 Feb 2024
class MovingAverageCalculator:
    def __init__(self, nSamples):
        self.nSamples = nSamples
        self.window = []
    
    def calculate_moving_average(self, x):
        self.window.append(x)

        if len(self.window) > self.nSamples:
            self.window.pop(0)

        average = sum(self.window) / len(self.window)
        return average


def draw(img, corners, imgpts):

    # why it has to be corners[1]?!
    corner = tuple(corners[1].ravel())
    
    # blue -- X
    img = cv2.line(img, (int(corner[0]),int(corner[1])), (int(imgpts[0][0][0]),int(imgpts[0][0][1])), (255, 0, 0), 5)
    # green -- Y 
    img = cv2.line(img, (int(corner[0]),int(corner[1])), (int(imgpts[1][0][0]),int(imgpts[1][0][1])), (0, 255, 0), 5)
    # red -- Z
    img = cv2.line(img, (int(corner[0]),int(corner[1])), (int(imgpts[2][0][0]),int(imgpts[2][0][1])), (0, 0, 255), 5)
    return img


def getOrientation(rvecs,tvecs):
    # converting Rodrigues format to 3x3 rotation matrix format
    R_c = cv2.Rodrigues(rvecs)[0]
    t_c = -np.matmul(R_c, tvecs)
    T_c = np.hstack([R_c, t_c])
    alpha, beta, gamma = cv2.decomposeProjectionMatrix(T_c)[-1]
    
    # Convert euler angles to roll-pitch-yaw of a camera with X forward and Y left
    roll = beta
    pitch = -alpha - 90
    yaw = gamma + 90

    return roll,pitch,yaw

def loadCalibrationData(yamlFileName):
    with open(yamlFileName, 'r') as file:
        data = yaml.safe_load(file)

    # Extract the camera matrix and distortion coefficients
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeff'])
    #print(camera_matrix,dist_coeffs)
    return camera_matrix,dist_coeffs

def pose_esitmation(frame, aruco_dict_type, mtx, dist):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    global prev_save_time, targetPosDepthCam, stacked_frame, debug_message

    # Define the font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.
    font_color = (0, 0, 255)  # red color
    line_type = cv2.LINE_AA

    # Write the debug message on the image
    
    text_size, _ = cv2.getTextSize("Debug Message", font, font_scale, 1)
    text_x = 10
    text_y = 10 + text_size[1]
    background_color = (50, 50, 50)  # dark grey background color

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    #parameters = cv2.aruco.DetectorParameters_create()

    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    # 3D object points, assuming the chessboard lies on the X-Y plane (z=0) and square size of 38mm
    square_size=38*2
    objp = np.zeros((4, 3), np.float32)
    #objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)*square_size
    objp[0]=[0.,square_size*1.,0.]
    objp[1]=[0.,0.,0.]
    objp[2]=[square_size*1.,0.,0.]
    objp[3]=[square_size*1.,square_size*1.,0.]

    axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)*square_size


    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            if ids[i][0]==590:
                
                
                # # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                # rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                #                                                            distortion_coefficients)
                # # Draw a square around the markers
                #cv2.aruco.drawDetectedMarkers(frame, [corners[i]]) 

                # # Draw Axis
                # cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

                cc = corners[i][0]
                cc = np.reshape(cc, (cc.shape[0], 1, 2))
                ret, rvecs, tvecs = cv2.solvePnP(objp,cc, mtx, dist)

                if ret:

                    roll,pitch,yaw = getOrientation(rvecs,tvecs)
                    
                    
                    debug_message='%.2f,%.2f,%.2f,(%d,%d),(%.1f,%.1f,%.1f)' % (tvecs[0]/1e3,tvecs[1]/1e3,tvecs[2]/1e3,int(cc[1][0][0]),int(cc[1][0][1]),roll,pitch,yaw)

                    # logging 
                    # Get the current system time
                    current_time = time.time()

                    if current_time - prev_save_time >= LogInterval:
                        logger.add_data('cam,%s' %(debug_message))
                        logger.add_data(targetPosDepthCam)
                        prev_save_time = current_time  # Update the previous save time

                    # Get the text size
                    (text_width, text_height), _ = cv2.getTextSize(debug_message, font, font_scale, 1)

                    # Calculate the background rectangle coordinates
                    background_rect_coords = ((0, 0), (text_width + 10, text_height + 10))  # Adjust padding as needed

                    # Draw the background rectangle
                    cv2.rectangle(frame, background_rect_coords[0], background_rect_coords[1], background_color, cv2.FILLED)
                    cv2.putText(frame, debug_message, (text_x, text_y), font, font_scale, font_color, 1, line_type)
                
                    
                    
                

                    # project 3D points to image plane
                    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

                    frame = draw(frame, cc, imgpts)
                    #cv2.imshow('Estimated Pose', img)


                    

    return frame

if __name__ == '__main__':


    opencv_version = cv2.__version__

    # make sure that opencv version is okay!!
    assert opencv_version == '4.7.0'

    ap = argparse.ArgumentParser()
    # ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    # ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument('-com','--com',type=str,required=True,help='comport name')
    ap.add_argument('-cam','--cam',type=str,required=True,help='camera index')
    ap.add_argument("-ip","--ip",type=str,required=True,help="remote rpi3 ip address")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    # get remote rpi3 ip address
    remoteHostIPAddr=args["ip"]
    
    cameraIndx = int(args['cam'])
    
    comportName= args['com']

    aruco_dict_type = ARUCO_DICT["DICT_ARUCO_ORIGINAL"]

    # calibration_matrix_path = args["K_Matrix"]
    # distortion_coefficients_path = args["D_Coeff"]
    
    # k = np.load(calibration_matrix_path)
    # d = np.load(distortion_coefficients_path)


    # initialize data logger

    # get current timestamp and generat the log file 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = DataLogger(log_interval=1, file_path="log/%s_data.log" %(timestamp))  # Specify the data file path
    loggerFPGA = DataLogger(log_interval=1, file_path="log/%s_fpga.log" %(timestamp))  # Specify the data file path
    loggerDebug = DataLogger(log_interval=1, file_path="log/%s_debug.log" %(timestamp))  # Specify the data file path
    
    # Start the logging process
    logger.start_logging()
    print('data logger started')
    logger.add_data('ISD Microphone Array System Started')

    loggerFPGA.start_logging()
    print('data logger (FPGA) started')
    loggerFPGA.add_data('ISD Microphone Array System (FPGA Communication) Started')

    loggerDebug.start_logging()
    print('data logger (debug) started')
    loggerDebug.add_data('ISD Microphone Array System (debug) Started')


    # load calibration data
    k,d = loadCalibrationData('20240226_1255_1m6_calibration_err0_05.yaml')


    # run depth camera 
    depthcam_thread = threading.Thread(target=depthcam_main, daemon=True)
    depthcam_thread.start()


    # if comportName not 'nil' and starts with 'COM'
    if comportName!='nil' and comportName.find('COM')==0:
        # start serial port thread
        ser = serial.Serial(comportName, 115200)

        # Start a separate thread to read data from the serial port
        stopSerialThread=False
        serial_thread = threading.Thread(target=read_serial_data, args=(ser,), daemon=True)
        serial_thread.start()

    # start thread to handle FPGA socket communication
    fpgaCommThread = threading.Thread(target=fpgaCommHandler,daemon=True)
    fpgaCommThread.start()


    video = cv2.VideoCapture(cameraIndx)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)
        key = cv2.waitKey(1) & 0xFF

        # press q or ESC to quit
        if key == ord('q') or (key%256)==27:
            break
        elif key == ord('n'):
            talkboxLocIndex = showDialogSelectTalkboxLocIndex()
            if talkboxLocIndex>=0 and talkboxLocIndex<30:
                set_testrig_XY_byIndex(remoteHostIPAddr,talkboxLocIndex)
                # log the event
                logger.add_data('talkbox location moved to index# %d' %(talkboxLocIndex))

        elif key == ord('m'):
            logger.add_data('positon changed!!')


    # end depthcam 
    endDepthCamFlag=True

    video.release()
    cv2.destroyAllWindows()

    # stop data logger
    logger.stop_logging()
    loggerFPGA.stop_logging()
    loggerDebug.stop_logging()

    # stop serial port thread
    stopSerialThread=True

    # close selectors
    sel.close()
    