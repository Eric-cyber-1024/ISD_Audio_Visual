'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100

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




# Add[support for pmd flexx2 depth camera],Brian,29 Feb 2024
K=5
DEPTH_CAM_WIDTH =224
DEPTH_CAM_HEIGHT=172

mouseX = 0
mouseY = 0


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
        M = self.cameraMatrix
        fx= M[0][0]
        fy= M[1][1]
        cx= M[0][2]
        cy= M[1][2]
        x = (u-cx)*(z/fx)
        y = (v-cy)*(z/fy)

        print(x,y,z)

    

    def onNewData(self, data):
        p = data.npoints()
        self.queue.put(p)
        

    def paint (self, data):
        """Called in the main thread, with data containing one of the items that was added to the
        queue in onNewData.
        """
        # mutex to lock out changes to the distortion while drawing
        self.lock.acquire()

        depth = data[:, :, 2]
        gray = data[:, :, 3]
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

        # equalize grayImage8
        grayImage8 = cv2.equalizeHist(grayImage8)    

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

        cv2.imshow('depth_plus_IR',stacked_frame)
        #cv2.imshow('Depth',zImg)
        #cv2.imshow('Gray', gImg)
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

        print(self.cameraMatrix)

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
        newGrayValue = clampedVal / 600 * 255
        return newGrayValue

def process_event_queue (q, painter):

    while True:
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
            # close if escape key pressed
            if currentKey == 27: 
                break

# depth camera main
def depthcam_main():

    # Set the available arguments
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser (usage = __doc__)
    add_camera_opener_options (parser)
    options = parser.parse_args()
    
    # for testing only
    options.rrf = 'meetingroom4.rrf'
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

    # Define the font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.
    font_color = (255, 255, 255)  # White color
    line_type = cv2.LINE_AA

    # Write the debug message on the image
    debug_message = "Debug Message"
    text_size, _ = cv2.getTextSize(debug_message, font, font_scale, 1)
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
    square_size=38
    objp = np.zeros((4, 3), np.float32)
    #objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)*square_size
    objp[0]=[0.,38.,0.]
    objp[1]=[0.,0.,0.]
    objp[2]=[38.,0.,0.]
    objp[3]=[38.,38.,0.]

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
                    
                    tvecs = tvecs*2.
                    debug_message='%.2f,%.2f,%.2f,(%d,%d),(%.1f,%.1f,%.1f)' % (tvecs[0]/1e3,tvecs[1]/1e3,tvecs[2]/1e3,int(cc[1][0][0]),int(cc[1][0][1]),roll,pitch,yaw)

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
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]

    # calibration_matrix_path = args["K_Matrix"]
    # distortion_coefficients_path = args["D_Coeff"]
    
    # k = np.load(calibration_matrix_path)
    # d = np.load(distortion_coefficients_path)


    # initialize data logger

    # get current timestamp and generat the log file 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = DataLogger(log_interval=1, file_path="log/%s_data.log" %(timestamp))  # Specify the data file path
    # Start the logging process
    logger.start_logging()
    print('data logger started')


    # load calibration data
    k,d = loadCalibrationData('20240226_1255_1m6_calibration_err0_05.yaml')


    # run depth camera 
    depthcam_thread = threading.Thread(target=depthcam_main, daemon=True)
    depthcam_thread.start()


    video = cv2.VideoCapture(1)
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

    video.release()
    cv2.destroyAllWindows()

    # stop data logger
    logger.stop_logging()