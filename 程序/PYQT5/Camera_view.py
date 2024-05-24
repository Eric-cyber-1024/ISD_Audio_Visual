#from pywinauto import Desktop  # add this to handle UI scaling issue
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
import queue

import sys
import cv2

from Style import *
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import wavio  #pip3 install wavio
from datetime import datetime

import time
from sys import exit

from Widget_library import *
from Event import *
from Enum_library import *
from Delay_Transmission import *
from audio_controller import AudioDeviceDialog, AudioController, MyAudioUtilities
from utility import audio_dev_to_str, networkController, rescale_frame
from data_logger import DataLogger
from test_delay_cal import *

# added[for d435]
import pyrealsense2 as rs

from progress_bar import CustomProgressBarLogger
from utils import ARUCO_DICT

# from pywinauto import Desktop
current_datetime = datetime.now()
# Format the date and time
formatted_datetime = current_datetime.strftime("%m-%d-%y")
# Define Directory Path
VIDEO_SAVE_DIRECTORY = "\Video"
AUDIO_PATH = "\Audio"
OUTPUT_PATH = "\Combined"
VIDEO_DATE = "\\" + formatted_datetime
AUDIO_NAME = ""
VIDEO_NAME = ""
OUTPUT_NAME = ""

display_monitor = 0
CURRENT_PATH = os.getcwd()
START_RECORDING = False
MIC_ON = True
SOUND_ON = True
DEBUG = False
ALIGNED_FRAMES = False
FILTERED_FRAMES = False
SENDING_PACKET = False
START_SEND_PACKET = True
MOUSE_CLICKED = False
sVersion='0.1.8'

def config_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath("./res")

    return os.path.join(base_path, relative_path)


def get_camera_list():
    # Get available camera indices
    available_cameras = []
    for i, camera in enumerate(QCameraInfo.availableCameras()):
        devLongName    = camera.deviceName()
        devDescription = camera.description()

        # check devLongName to find D455 RGB Camera as it appeared as "USB Video Camera" instead sometimes
        if devLongName.find('vid_8086&pid_0b5c&mi_03')>0:
            devDescription = 'Intel(R) RealSense(TM) Depth Camera 455  RGB'
        available_cameras.append(devDescription)
    return available_cameras

class ClickableLabel(QLabel):
    '''
    a clickable QLabel, if clicked 7 times within self.timeoutVal, a signal will be emitted
    '''
    clicked = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.counter = 0
        self.timeoutVal=2000 # timeout in 2 seconds
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.handleTimeout)

    def mousePressEvent(self, event):
        # print(self.counter)
        if self.counter==0:
            # start a timer
            self.timer.start(self.timeoutVal) 

        self.counter += 1
        #print(self.counter)
        if self.counter == 7:
            self.counter=0
            self.timer.stop()
            self.clicked.emit()
                

    def handleTimeout(self):
        #print('timeout')
        self.counter=0
    

class WorkerTryPing(QObject):
    finished = pyqtSignal(tuple)
    progress = pyqtSignal(int)

    def setHostIP(self,hostIP):
        self.hostIP = hostIP

    def run(self):
        myNetworkController = networkController()
        #myNetworkController.list_network_adapters()
        results = myNetworkController.tryPing(self.hostIP)
        self.finished.emit(results)
        

class EchoText(QWidget): 

    def test(self):
        print('hihi')
  
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.layout = QGridLayout() 
        self.setLayout(self.layout) 
  
        points = [(50, 50), (100, 100), (150, 150), (200, 200)]


        #self.canvas = CanvasWidget(points, self.test)

        self.textbox = QLineEdit() 
        self.echo_label = QLabel('') 
  
        self.textbox.textChanged.connect(self.textbox_text_changed) 
  
        self.layout.addWidget(self.textbox,0,0,1,1) 
        self.layout.addWidget(self.echo_label,1,0,1,1) 
        #self.layout.addWidget(self.canvas,1,0)
  
    def textbox_text_changed(self): 
        self.echo_label.setText(self.textbox.text()) 


class PointSelectionGUI(QWidget):
    def __init__(self, points, callback):
        super().__init__()

        self.canvas = CanvasWidget(points, callback)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.setWindowTitle("Point Selection GUI")

class CanvasWidget(QWidget):
    def __init__(self, points, callback):
        super().__init__()

        self.points = points
        self.callback = callback

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.black)

        for i, (x, y) in enumerate(self.points):
            painter.setBrush(Qt.red)
            painter.drawEllipse(x - 5, y - 5, 10, 10)

            if i < 10:
                painter.setPen(Qt.black)
                painter.drawText(x, y - 10, f"M0{i}")
            else:
                painter.setPen(Qt.black)
                painter.drawText(x, y - 10, f"M{i}")

        painter.end()

    def mousePressEvent(self, event):
        x, y = event.x(), event.y()

        for i, (px, py) in enumerate(self.points):
            if abs(px - x) <= 5 and abs(py - y) <= 5:
                self.callback(str(i))
                break


# Add[for getting moving averages],Brian,1 April 2024
class MovingAverageCalculator:
    
    cal_timeout = pyqtSignal()

    def __init__(self, nSamples):
        self.nSamples = nSamples
        self.window = []
        self.locked = 0
        self.lock_var_range = 0.05
        self.locked_value = 0.0
        self.average = 0.0

    def calculate_moving_average(self, x):
        self.window.append(x)

        if len(self.window) > self.nSamples:
            self.window.pop(0)

        self.average = sum(self.window) / len(self.window)
        return self.average

    # Add lock value, Jason, 10 April 2024
    def calculate_moving_average_lock(self, x):
        '''
        Save the averaged depth value as locked_value if 
        the variance of the data in the window is smaller than
        lock_var_range
        '''
        self.window.append(x)

        if len(self.window) > self.nSamples:
            self.window.pop(0)
        
        self.average = sum(self.window) / len(self.window)

        # print('var: ', np.var(self.window))
        if np.var(self.window) < self.lock_var_range:
            self.locked = 1
        else:
            self.locked = 0

        if self.locked:
            self.locked_value = self.average
            return self.locked_value
        
        return 0.0

    def isDepthLocked(self):
        return self.locked
    

class WebCam(QThread):
    '''
    class to get access to web camera data
    '''
    COLOR_CAM_WIDTH  = 1920#1920
    COLOR_CAM_HEIGHT = 1080#1080
    COLOR_FPS        = 15   # have to reduced to 15 on Surface Pro 9

    def __init__(self, camera_index, display_width, display_height):
        super().__init__()

        self.is_paused = False

        self.camera_index = camera_index
        self.DISPLAY_WIDTH = display_width
        self.DISPLAY_HEIGHT = display_height

        self.time_per_frame = 1.0/WebCam.COLOR_FPS
        self.fps_cnt = 0

        self.q_color_frame = queue.Queue()
        self.q_frame_output = queue.Queue()

        self.cap = cv2.VideoCapture(self.camera_index)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.DISPLAY_WIDTH)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.DISPLAY_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.COLOR_CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.COLOR_CAM_HEIGHT)

        # print(self.cap.get(3), self.cap.get(4))
        # video_width = int(self.cap.get(3))
        # video_height = int(self.cap.get(4))
        # print("Input ratio w-h: ",video_width, video_height)

    def setMousePixel(self, mousex, mousey):
        if self.DISPLAY_HEIGHT != d435.COLOR_CAM_HEIGHT or self.DISPLAY_WIDTH != d435.COLOR_CAM_WIDTH:
            mousex = mousex * d435.COLOR_CAM_WIDTH / self.DISPLAY_WIDTH
            mousey = mousey * d435.COLOR_CAM_HEIGHT / self.DISPLAY_HEIGHT
        
        self.i = mousex
        self.j = mousey

    def readFrameFromDisplayQueue(self):
        if self.q_color_frame.empty():
            return True, {'frame': None}

        # drop frames
        q_size = self.q_color_frame.qsize()
        if q_size > 1:
            for i in range(q_size//2):
                self.q_color_frame.get()

        return False, self.q_color_frame.get()

    def readFrameFromOutputQueue(self):
        if self.q_frame_output.empty():
            return True, {'frame': None, 'timestamp': None}

        # drop frames
        q_size = self.q_color_frame.qsize()
        if q_size > 1:
                self.q_color_frame.get()

        return False, self.q_frame_output.get()

    def pause_thread(self):
        self.is_paused = True

    def resume_thread(self):
        self.prev_time = time.time()
        self.is_paused = False

    def run(self):
        global START_RECORDING
        self.prev_time = time.time()
        prev_time_second = time.time()  # previous time for fps count
        while True:
            if self.is_paused:
                self.sleep(5)
                continue

            try:
                time_elapsed = time.time() - self.prev_time
                _, cv_img = self.cap.read()

                cur_time = time.time()
                if time_elapsed > self.time_per_frame:
                    self.prev_time += self.time_per_frame
                    self.q_color_frame.put({'frame': cv_img})

                    if START_RECORDING:
                        self.q_frame_output.put({'frame': cv_img, 'timestamp': time.time()})
                    self.fps_cnt += 1

                if cur_time - prev_time_second > 1.0:
                    print('CamFPS: ', self.fps_cnt)
                    # print('Frame QSize: ', self.q_color_frame.qsize())
                    # print('Out QSize: ', self.q_frame_output.qsize())
                    prev_time_second = cur_time
                    self.fps_cnt = 0
            except:
                print('getFrame Failed')


class d435(QThread):
    '''
    class to get access to d435 data
    '''
    frame_ready = pyqtSignal(tuple)

    DEPTH_CAM_WIDTH  = 1280#848#1280
    DEPTH_CAM_HEIGHT = 720#480#720
    DEPTH_FPS        = 15

    COLOR_CAM_WIDTH  = 1280#1920
    COLOR_CAM_HEIGHT = 720#1080
    COLOR_FPS        = 15#15   # have to reduced to 15 on Surface Pro 9

    def __init__(self, display_width, display_height):
        super().__init__()

        self.is_paused = False

        self.DISPLAY_WIDTH = display_width
        self.DISPLAY_HEIGHT = display_height

        # queue for color frames, Jason, 7 May 2024
        self.q_color_frame = queue.Queue()  # RGB frame for display, {'frame':,'point':,'timestamp':}
        self.q_frame_output = queue.Queue() # RGB frame for output, {'frame':,'timestamp':}

        # initialize the moving average calculator, window size =16 samples
        self.ma = MovingAverageCalculator(nSamples=16)

        # Create a pipeline object to manage streaming
        self.pipeline = rs.pipeline()

        # Create a config object to configure the streams
        self.config = rs.config()

        # Enable the depth and color streams.
        self.config.enable_stream(rs.stream.depth, d435.DEPTH_CAM_WIDTH, d435.DEPTH_CAM_HEIGHT,
                            rs.format.z16, d435.DEPTH_FPS)
        self.config.enable_stream(rs.stream.color, d435.COLOR_CAM_WIDTH, d435.COLOR_CAM_HEIGHT,
                            rs.format.bgr8, d435.COLOR_FPS)

        # Start the pipeline streaming
        profile = self.pipeline.start(self.config)

        # Create an align object to align the depth and color frames
        self.align = rs.align(rs.stream.color)

        # Get the intrinsics of the color camera
        self.colorProfile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        self.colorIntrinsics = self.colorProfile.get_intrinsics()
        print("colorIntrinsics: ", self.colorIntrinsics)
        # Get the intrinsics of the depth camera
        self.depthProfile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        self.depthIntrinsics = self.depthProfile.get_intrinsics()
        # print("depthIntrinsics: ", self.depthIntrinsics)
        w, h = self.depthIntrinsics.width, self.depthIntrinsics.height

        # Get the depth scale of the depth sensor
        self.depthSensor = profile.get_device().first_depth_sensor()
        self.depthScale  = self.depthSensor.get_depth_scale()
        # print("depthScale: ", self.depthScale)

        # Get extrinsics
        self.depthToColorExtrinsics = self.depthProfile.get_extrinsics_to(self.colorProfile)
        self.colorToDepthExtrinsics = self.colorProfile.get_extrinsics_to(self.depthProfile)

        # for visualization
        self.depthMin = 0.1  #meter
        self.depthMax = 15.0  #meter

        # 2D image coordinate
        self.i=0  # from mouse x
        self.j=0  # from mouse y

        # previous mouse x, y
        self.iPrev=self.i
        self.jPrev=self.j

        self.point = [-1.,-1.,-1]

        # # Print Visual Preset Information
        # depth_sensor = profile.get_device().first_depth_sensor()
        # preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        # print('preset range: ', preset_range)
        # for i in range(int(preset_range.max)):
        #     visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
        #     print('%02d: %s'%(i,visulpreset))

        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.visual_preset,1)  # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
                                                              # 0=Custom, 1=Default, 2=Hand, 3=High Accuracy, 4=High Density
        self.colorizer.set_option(rs.option.min_distance, self.depthMin)
        self.colorizer.set_option(rs.option.max_distance, self.depthMax)

    def getOpencvIntrinsics(self):
        '''
        get color frame intrinsics in Opencv Format

        '''
        # get self.k, self.d from self.d435 color intrinsic matrix
        intrinsics = self.colorIntrinsics

        fx = intrinsics.fx  # Focal length in x
        fy = intrinsics.fy  # Focal length in y
        cx = intrinsics.ppx # Principal point x
        cy = intrinsics.ppy # Principal point y

        
        distortion_coeffs = [[intrinsics.coeffs[0], intrinsics.coeffs[1], intrinsics.coeffs[4],
                     intrinsics.coeffs[3], intrinsics.coeffs[2]]]
        
        print(distortion_coeffs)

        camera_matrix = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])
        dist_coeffs = np.array(distortion_coeffs)
        print(camera_matrix,dist_coeffs)

        return camera_matrix,dist_coeffs

    # Revised, Jason, 14 May 2024
    def setMousePixel(self, mousex, mousey):
        '''
        mousex, mousey - mouse click coordinates on display (can be scaled up)
        self.i, self.j - mouse click coordinates on color frame
        '''
        if self.DISPLAY_HEIGHT != d435.COLOR_CAM_HEIGHT or self.DISPLAY_WIDTH != d435.COLOR_CAM_WIDTH:
            mousex = mousex * d435.COLOR_CAM_WIDTH / self.DISPLAY_WIDTH
            mousey = mousey * d435.COLOR_CAM_HEIGHT / self.DISPLAY_HEIGHT
        
        self.i = mousex
        self.j = mousey

    # Add for reading queue, Jason, 7 May 2024
    def readFrameFromDisplayQueue(self):
        if self.q_color_frame.empty():
            return True, {'frame': None, 'point': None, 'timestamp': None}

        # drop frames
        q_size = self.q_color_frame.qsize()
        if q_size > 1:
            for i in range(q_size//2):
                self.q_color_frame.get()
            
        return False, self.q_color_frame.get()

    def readFrameFromOutputQueue(self):
        if self.q_frame_output.empty():
            return True, {'frame': None, 'timestamp': None}

        # drop frames
        q_size = self.q_color_frame.qsize()
        if q_size > 1:
                self.q_color_frame.get()

        return False, self.q_frame_output.get()
    
    # Revised, Jason, 14 May 2024
    def getFrame(self):
        global START_RECORDING, ALIGNED_FRAMES

        # Get a frameset from the pipeline
        try:
            frameset = self.pipeline.wait_for_frames()

            # Get the aligned depth and color frames
            depthFrame = frameset.get_depth_frame()
            colorFrame = frameset.get_color_frame()
            
            # Validate that both frames are valid
            if not depthFrame or not colorFrame:
                return None,None,None

            # Convert the images to numpy arrays
            depthFrameData=depthFrame.get_data()
            depthImage = np.asanyarray(depthFrameData)
            colorImage = np.asanyarray(colorFrame.get_data())
            
            if START_RECORDING:
                self.q_frame_output.put({'frame': colorImage, 'timestamp': time.time()})
            
            
            # project color pixel to depth pixel
            # note that self.i,self.j are mouse x,y from 1920x1020 displayed frame divided by 1.5
            # assuming colorFrame is a 1280x720 frame !!
            depthPixel = rs.rs2_project_color_pixel_to_depth_pixel(
                depthFrameData, self.depthScale, self.depthMin,
                self.depthMax, self.depthIntrinsics, self.colorIntrinsics,
                self.depthToColorExtrinsics, self.colorToDepthExtrinsics, [self.i, self.j])
            
            if depthPixel[0]>=0 and depthPixel[1]>=0:
                depth = depthFrame.get_distance(int(depthPixel[0]),int(depthPixel[1]))
                depth = self.ma.calculate_moving_average_lock(depth)

                # project depth pixel to 3D point
                # x is right+, y is down+, z is forward+
                self.point = rs.rs2_deproject_pixel_to_point(self.depthIntrinsics,[int(depthPixel[0]), int(depthPixel[1])], depth)

                x = self.point[0]
                y = self.point[1]
                z = self.point[2]
                # print(self.i,self.j,depthPixel,self.point)

            # update self.iPrev,jPrev
            self.iPrev = self.i
            self.jPrev = self.j


            # try to add aligned frame,Brian,23 May 2024
            if ALIGNED_FRAMES:

                if FILTERED_FRAMES:
                    filtered_depth_frame = depthFrame
                    dec_filter = rs.decimation_filter()
                    depth_to_disparity = rs.disparity_transform(True)
                    spat_filter = rs.spatial_filter()
                    temp_filter = rs.temporal_filter()
                    disparity_to_depth = rs.disparity_transform(False)
                    hole_filling = rs.hole_filling_filter()

                    # filtered_depth_frame = dec_filter.process(filtered_depth_frame)
                    filtered_depth_frame = depth_to_disparity.process(filtered_depth_frame)
                    filtered_depth_frame = spat_filter.process(filtered_depth_frame)
                    filtered_depth_frame = temp_filter.process(filtered_depth_frame) #need more frames
                    filtered_depth_frame = disparity_to_depth.process(filtered_depth_frame)
                    filtered_depth_frame = hole_filling.process(filtered_depth_frame)


                depth_colormap = np.asanyarray(self.colorizer.colorize(depthFrame).get_data())
                alpha = 0.5
                colorImage = cv2.addWeighted(depth_colormap, alpha, colorImage, 1 - alpha, 0 )

            # Revised to use queue, Jason, 7 May 2024
            self.q_color_frame.put({'frame': colorImage, 'point':self.point, 'timestamp': time.time()})
        except:
            print("getFrame Failed")

    # Revised, Jason - 17 May 2024 - Aligned Frames
    def getFrameWithAlignedFrames(self):
        global FILTERED_FRAMES, START_RECORDING

        # self.i = mousex
        # self.j = mousey
        # print(self.i,self.j) # for debugging
    
        # Get a frameset from the pipeline
        try:
            frameset = self.pipeline.wait_for_frames()

            # Align the depth frame to the color frame
            aligned_frameset = self.align.process(frameset)
       
            depth_frame = aligned_frameset.get_depth_frame()
            color_frame = aligned_frameset.get_color_frame()

            # Convert the images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            if START_RECORDING:
                self.q_frame_output.put({'frame': color_image, 'timestamp': time.time()})

            # # test filters for depth frame, Jason, 12 April 2024
            # Using default value
            # Depth Frame >> Decimation Filter >> Depth2Disparity Transform** 
            # -> Spatial Filter >> Temporal Filter >> Disparity2Depth Transform** 
            # >> Hole Filling Filter >> Filtered Depth.
            if FILTERED_FRAMES:
                filtered_depth_frame = depth_frame
                dec_filter = rs.decimation_filter()
                depth_to_disparity = rs.disparity_transform(True)
                spat_filter = rs.spatial_filter()
                temp_filter = rs.temporal_filter()
                disparity_to_depth = rs.disparity_transform(False)
                hole_filling = rs.hole_filling_filter()

                # filtered_depth_frame = dec_filter.process(filtered_depth_frame)
                filtered_depth_frame = depth_to_disparity.process(filtered_depth_frame)
                filtered_depth_frame = spat_filter.process(filtered_depth_frame)
                filtered_depth_frame = temp_filter.process(filtered_depth_frame) #need more frames
                filtered_depth_frame = disparity_to_depth.process(filtered_depth_frame)
                filtered_depth_frame = hole_filling.process(filtered_depth_frame)

            # Validate that both frames are valid
            if not depth_frame or not color_frame:
                return

            # # project color pixel to depth pixel, need to check as it is not accurate for same resolution
            depthPixel = rs.rs2_project_color_pixel_to_depth_pixel(
                depth_frame.get_data(), self.depthScale, self.depthMin,
                self.depthMax, self.depthIntrinsics, self.colorIntrinsics,
                self.depthToColorExtrinsics, self.colorToDepthExtrinsics, [self.i, self.j])
            # depthPixel = [self.i, self.j]

            if depthPixel[0]>=0 and depthPixel[1]>=0:
                depth = depth_frame.get_distance(int(depthPixel[0]),int(depthPixel[1]))

                # Add lock value, Jason, 10 April 2024
                depth = self.ma.calculate_moving_average_lock(depth)

                # project depth pixel to 3D point
                # x is right+, y is down+, z is forward+
                self.point = rs.rs2_deproject_pixel_to_point(self.depthIntrinsics,[int(depthPixel[0]), int(depthPixel[1])], depth)

                x = self.point[0]/1.1
                y = self.point[1]/1.1
                z = self.point[2]/1.1

            # update self.iPrev,jPrev
            self.iPrev = self.i
            self.jPrev = self.j

            # add depth color to frame
            depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            alpha = 0.5
            if FILTERED_FRAMES:
                if depth_colormap.shape[0]<720:
                    depth_colormap = rescale_frame(depth_colormap, color_image.shape[1], color_image.shape[0])
            
            color_image = cv2.addWeighted(depth_colormap, alpha, color_image, 1 - alpha, 0 )
            self.q_color_frame.put({'frame': color_image, 'point':self.point, 'timestamp': time.time()})
        except:
            print("getFrameWithAlignedFrames Failed")

    def pause_thread(self):
        self.is_paused = True

    def resume_thread(self):
        self.is_paused = False

    # Get frames continuously in thread, Jason, 14 May 2024
    def run(self):
        global ALIGNED_FRAMES

        fps_cnt = 0
        cur_time = time.time()
        while True:
            if self.is_paused:
                self.sleep(5)   # sleep when combining video and audio
                continue

            # revised[keep using getFrame to handle showing aligned frame or not],Brian,23 May 2024
            # if ALIGNED_FRAMES:
            #     self.getFrameWithAlignedFrames()
            # else:
            #     self.getFrame()
            self.getFrame()
            fps_cnt += 1
            if time.time() - cur_time > 1.0:
                print('Cam FPS: ', fps_cnt)
                # print('Frame QSize: ', self.q_color_frame.qsize())
                # print('Out QSize: ', self.q_frame_output.qsize())
                fps_cnt = 0
                cur_time = time.time()
            # else:
                # self.usleep(1000000//self.COLOR_FPS//2)


class CameraSelectionDialog(QDialog):

    def __init__(self, camera_names):
        super().__init__()

        self.setWindowTitle("Camera Selection")
        self.setFixedWidth(1000)
        self.setFixedHeight(200)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(
                resource_path("mic_double_FILL0_wght400_GRAD0_opsz24.svg")),
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.lbl_dialogTitle = QLabel('Camera Selection')
        self.lbl_dialogTitle.setStyleSheet(LABEL_STYLE_CAM_DIAG)
        self.camera_combo = QComboBox()
        self.camera_combo.setStyleSheet(COMBO_STYLE_CAM_DIAG)
        self.camera_combo.addItems(camera_names)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok
                                      | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.setStyleSheet(BUTTON_STYLE_CAM_DIAG)

        layout = QVBoxLayout()
        layout.addWidget(self.lbl_dialogTitle)
        layout.addWidget(self.camera_combo)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_selected_camera_name(self):
        return self.camera_combo.currentIndex(), self.camera_combo.currentText()


# Combine As output thread
class VideoAudioThread(QThread):
    start_writing = pyqtSignal()
    # finished = pyqtSignal()

    def __init__(self, video_path, audio_path, output_path, logger):
        super().__init__()
        self.video_path = video_path
        self.audio_path = audio_path
        self.output_path = output_path
        self.logger = logger

    def run(self):
        print("Combine Video")
        time.sleep(1)
        self.start_writing.emit()
        print('v: ', self.video_path)
        print('a: ', self.audio_path)
        print('o: ', self.output_path)
        combine_video_audio(self)


# Video Thread
class VideoThread(QThread):
    depth_value_locked = pyqtSignal()
    update_3d_coordinate = pyqtSignal(list)
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_index, camera_name, display_width, display_height, usePoseEstimation=False):
        super().__init__()
        self.camera_index = camera_index
        self.camera_name  = camera_name
        self.d435 = None    # Intel RealSense D435
        self.cam = None     # Generic camera without depth cam
        self.mousex=0
        self.mousey=0
        self.DISPLAY_WIDTH = display_width
        self.DISPLAY_HEIGHT = display_height
        self.is_paused = False
        self.usePoseEstimation=usePoseEstimation

        self.aruco_dict_type = ARUCO_DICT["DICT_ARUCO_ORIGINAL"]
        

        if self.camera_name.startswith('Intel(R) RealSense(TM) Depth Camera 4') and self.camera_name.endswith('RGB'):
            self.d435 = d435(self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT)
            self.k, self.d = self.d435.getOpencvIntrinsics()
        else:
            self.cam = WebCam(self.camera_index, self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT)

    def setMouseXY(self,mousex,mousey):
        self.mousex = mousex
        self.mousey = mousey
        if self.d435:
            self.d435.setMousePixel(mousex, mousey)
        if self.cam:
            self.cam.setMousePixel(mousex, mousey)

    def drawDebugText(self,sMsg):
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale= 1.2
        background_color = (50, 50, 50)  # dark grey background color
        textColor = (0, 0, 255)
        lineType = cv2.LINE_AA
        
        # Get the text size
        (text_width, text_height), _ = cv2.getTextSize(sMsg, fontFace, fontScale, 1)

        text_x = 10
        text_y = 10+text_height

        # Calculate the background rectangle coordinates
        background_rect_coords = ((0, 0), (text_width + 10, text_height + 20))  # Adjust padding as needed

        cv2.rectangle(self.cv_img, background_rect_coords[0], background_rect_coords[1], background_color, cv2.FILLED)
        cv2.putText(self.cv_img, sMsg, (text_x, text_y), fontFace, fontScale, textColor, 2, lineType)

    # def rescale_frame(self,frame, target_width, target_height):
    #     # width  = int(frame.shape[1] * percent/ 100)
    #     # height = int(frame.shape[0] * percent/ 100)
    #     width  = int(frame.shape[1] * target_width/ frame.shape[1])
    #     height = int(frame.shape[0] * target_height/ frame.shape[0])
    #     dim = (width, height)
    #     return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

    def getOrientation(self,rvecs,tvecs):
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
    
    def loadCalibrationData(self,yamlFileName):
        '''
        try to load camera intrinsic matrix from yaml file
        '''
        try:
            with open(yamlFileName, 'r') as file:
                data = yaml.safe_load(file)

                # Extract the camera matrix and distortion coefficients
                camera_matrix = np.array(data['camera_matrix'])
                dist_coeffs = np.array(data['dist_coeff'])
                #print(camera_matrix,dist_coeffs)
                return camera_matrix,dist_coeffs
        except:
            return None, None

    def pose_estimation(self,frame, aruco_dict_type, mtx, dist):

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
        text_y = 40 + text_size[1]
        background_color = (50, 50, 50)  # dark grey background color

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
        #parameters = cv2.aruco.DetectorParameters_create()

        arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
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

                        roll,pitch,yaw = self.getOrientation(rvecs,tvecs)
                        
                        
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

                        frame = self.draw(frame, cc, imgpts)
                        #cv2.imshow('Estimated Pose', img)


                        

        return frame
    

    
    def draw(self,img, corners, imgpts):
        '''
        draw bounding box for 3d pose estimation
        '''
        # why it has to be corners[1]?!
        corner = tuple(corners[1].ravel())
        
        # blue -- X
        img = cv2.line(img, (int(corner[0]),int(corner[1])), (int(imgpts[0][0][0]),int(imgpts[0][0][1])), (255, 0, 0), 5)
        # green -- Y 
        img = cv2.line(img, (int(corner[0]),int(corner[1])), (int(imgpts[1][0][0]),int(imgpts[1][0][1])), (0, 255, 0), 5)
        # red -- Z
        img = cv2.line(img, (int(corner[0]),int(corner[1])), (int(imgpts[2][0][0]),int(imgpts[2][0][1])), (0, 0, 255), 5)
        return img

    def pause_thread(self):
        self.is_paused = True

    def resume_thread(self):
        print('resume')
        self.is_paused = False

    def run(self):
        global MOUSE_CLICKED, START_SEND_PACKET

        while True:
            if self.is_paused:
                self.sleep(5)
                continue

            if self.d435:
                is_empty, encoded_frame = self.d435.readFrameFromDisplayQueue()

                if is_empty:
                    self.msleep(15) # sleep for getting frames
                    continue

                point = encoded_frame['point']
                self.cv_img = encoded_frame['frame']

                if point is not None:
                    # emit signal 
                    self.update_3d_coordinate.emit(point)

                # send packet when depth locked, Jason, 16 April 2024
                if START_SEND_PACKET and MOUSE_CLICKED and self.d435.ma.isDepthLocked():
                    MOUSE_CLICKED = False
                    self.depth_value_locked.emit()

                # try to perform 3d pose estimation here
                if self.usePoseEstimation:
                    self.cv_img = self.pose_estimation(self.cv_img, self.aruco_dict_type, self.k, self.d)
                

                if self.cv_img is None:
                    ret = False
                    self.usleep(1000000//self.d435.COLOR_FPS)
                else:
                    # add scale self.cv_img if it's not 1080p
                    if self.cv_img.shape[0]<1080:
                        # self.cv_img = self.rescale_frame(self.cv_img,150)
                        self.cv_img = rescale_frame(self.cv_img, 1920, 1080)

                    ret = True

            elif self.cam:
                is_empty, encoded_frame = self.cam.readFrameFromDisplayQueue()

                if is_empty:
                    self.msleep(15)
                    continue

                self.cv_img = encoded_frame['frame']
                if self.cv_img is None:
                    ret = False
                    self.usleep(1000000//self.cam.COLOR_FPS)
                else:
                    if self.cv_img.shape[0]<1080:
                        # self.cv_img = self.rescale_frame(self.cv_img, 150)
                        self.cv_img = rescale_frame(self.cv_img, 1920, 1080)

                    ret = True

            if ret:
                if DEBUG:
                    yInterval = self.DISPLAY_HEIGHT//4
                    xInterval = self.DISPLAY_WIDTH//4
                    gridColor = (255,100,15)

                    if self.camera_name.startswith('Intel(R) RealSense(TM) Depth Camera 4') and self.camera_name.endswith('RGB'):
                        if point is not None:
                            # draw debug texts on top-left corner
                            pointStr = 'x:%.2f,y:%.2f,z:%.2f' % (point[0], point[1], point[2])
                            self.drawDebugText(pointStr)

                            # draw ROI of the mouse x,y
                            cv2.rectangle(self.cv_img,(self.mousex-50,self.mousey-50),(self.mousex+50,self.mousey+50),(0,0,255),2)
                            cv2.drawMarker(self.cv_img,(self.mousex,self.mousey),(0, 0, 255),cv2.MARKER_CROSS,20,3)
                    
                    # veritical grid lines
                    cv2.line(self.cv_img, (xInterval, 0),  (xInterval, self.DISPLAY_HEIGHT),  gridColor, 2)
                    cv2.line(self.cv_img, (2*xInterval, 0),(2*xInterval, self.DISPLAY_HEIGHT),gridColor, 2)
                    cv2.line(self.cv_img, (3*xInterval, 0),(3*xInterval, self.DISPLAY_HEIGHT),gridColor, 2)

                    # horizontal grid lines
                    cv2.line(self.cv_img, (0, yInterval),  (self.DISPLAY_WIDTH, yInterval),  gridColor, 2)
                    cv2.line(self.cv_img, (0, 2*yInterval),(self.DISPLAY_WIDTH, 2*yInterval),gridColor, 2)
                    cv2.line(self.cv_img, (0, 3*yInterval),(self.DISPLAY_WIDTH, 3*yInterval),gridColor, 2)

                self.change_pixmap_signal.emit(self.cv_img)



class VideoSavingThread(QThread):
    def __init__(self, cam, d435, video_width, video_height):
        global VIDEO_NAME, OUTPUT_NAME
        super().__init__()
        self.cam = cam
        self.d435 = d435
        self.video_width = video_width
        self.video_height = video_height
        self.is_initialized = False
        print('Initiate Video')
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime(
            "[%m-%d-%y]%H_%M_%S")
        self.video_name = CURRENT_PATH + VIDEO_SAVE_DIRECTORY + VIDEO_DATE + "\\" + str(
            formatted_datetime) + '.avi'
        self.output_path = CURRENT_PATH + OUTPUT_PATH + VIDEO_DATE + "\\" + str(
            formatted_datetime) + '.mp4'
        VIDEO_NAME = self.video_name
        OUTPUT_NAME = self.output_path
        print('video name: ', self.video_name)
        print('combined video: ', self.output_path)
        if self.d435:
            self.video_width = self.d435.COLOR_CAM_WIDTH
            self.video_height = self.d435.COLOR_CAM_HEIGHT
            self.FPS = self.d435.COLOR_FPS
        elif self.cam:
            self.video_width = self.cam.COLOR_CAM_WIDTH
            self.video_height = self.cam.COLOR_CAM_HEIGHT
            self.FPS = self.cam.COLOR_FPS

        self.out = cv2.VideoWriter(
            self.video_name,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.FPS,
            (self.video_width, self.video_height))

        self.test_fps_cnt = 0   # real fps
        self.out_fps_cnt = 0    # output fps - fixed
        self.total_f_cnt = 0    # total number of frames

    def run(self):
        global START_RECORDING

        while True:
            if self.d435:
                is_empty, encoded_frame = self.d435.readFrameFromOutputQueue()
            elif self.cam:
                is_empty, encoded_frame = self.cam.readFrameFromOutputQueue()
            
            if is_empty:
                if not START_RECORDING: #finished recording
                    break
                else:
                    self.msleep(15) #wait for next frame
                    continue

            frame = encoded_frame['frame']
            frame_timestamp = encoded_frame['timestamp']

            if not self.is_initialized:
                self.video_start_time = frame_timestamp
                self.prev_end_time = frame_timestamp
                self.is_initialized = True

            self.video_end_time = frame_timestamp
            if frame_timestamp - self.prev_end_time >= 1.0:
                # start next loop
                fps_dif = self.FPS - self.out_fps_cnt
                # print('Real FPS: ', self.test_fps_cnt)

                if fps_dif > 0:     # have less frames then expected
                    for j in range(fps_dif):
                        self.out.write(frame)
                        self.total_f_cnt += 1
                        self.out_fps_cnt += 1
                        # self.test_fps_cnt += 1
                    # print('out fps: ', self.out_fps_cnt)
                    self.out.write(frame)
                    self.out_fps_cnt = 1
                    self.test_fps_cnt = 1
                    self.total_f_cnt += 1
                    # print('FPS dropped')
                elif fps_dif == 0:  # match FPS or have more frames
                    self.out.write(frame)
                    self.out_fps_cnt = 1
                    self.test_fps_cnt = 1
                    self.total_f_cnt += 1
                    # print('FPS matched')

                self.prev_end_time += 1.0
                
            else:
                # write frame if within 1 second interval and count not exceed FPS
                if self.out_fps_cnt < self.FPS:
                    self.out.write(frame)
                    self.out_fps_cnt += 1
                    self.total_f_cnt += 1
                self.test_fps_cnt += 1

        if not START_RECORDING:
            self.out.release()
            print('total time: ', self.video_end_time - self.video_start_time)
            print('total frame count: ', self.total_f_cnt)
            self.out_fps_cnt = 0


# Audio Thread
class AudioThread(QThread):

    def __init__(self, input_device):
        super().__init__()
        self.sample_rate = 48000    # Hz
        self.audio_buffer = []      # Buffer to store recorded audio data
        self.moving_window= []      # a moving window to get average audio level
        self.input_device = input_device # the input device index
        self.avgLevel = 0.
        self.q = queue.Queue()      # queue to save input stream data
            
    def setRecordTime(self):
        self.current_datetime = datetime.now()
        self.formatted_datetime = current_datetime.strftime("[%m-%d-%y]%H_%M_%S")

    def getLevel(self):
        return np.abs(self.avgLevel)

    def run(self):
        global START_RECORDING, AUDIO_NAME

        # Use a callback function for non-blocking audio recording
        def callback(indata, frames, time, status):
            if status:
                print(status)
            #print("Recording audio...")
            if START_RECORDING:
                # save indata copy to q
                self.q.put(indata.copy())

        
        if START_RECORDING:
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("[%m-%d-%y]%H_%M_%S")
            subtype = 'PCM_16'
            dtype = 'int16' 
            audio_name = CURRENT_PATH + AUDIO_PATH + VIDEO_DATE + "\\" + str(formatted_datetime) + '.wav'
            AUDIO_NAME = audio_name
            print('audio file::',audio_name)
            with sf.SoundFile(audio_name, mode='w', subtype=subtype,samplerate=self.sample_rate, channels=1) as file:
                print('starting soundfile',file)
                with sd.InputStream(samplerate=self.sample_rate, dtype=dtype, channels=1, callback=callback):
                    while START_RECORDING:
                        try:
                            file.write(self.q.get(timeout=0.3))
                        except:
                            break
                    print('closing soundfile')
                    file.close()


class App(QWidget):

    CAM_DISPLAY_WIDTH = 1920
    CAM_DISPLAY_HEIGHT = 1080

    def __init__(self):
        global DEBUG
        super().__init__()

        # try to load configurations from yaml file (if the config.yaml exists)
        self.configParams = self.tryLoadConfig()
        print(self.configParams)
        DEBUG = self.configParams['debug']

        if (QDesktopWidget().screenCount() > 1):
            self.screen_number = 1
        else:
            self.screen_number = 0
        self.setWindowTitle("ISD Mic Array Control Panel — v%s" %(sVersion))
        #self.setStyleSheet("background-color:gray")

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(
                resource_path("mic_double_FILL0_wght400_GRAD0_opsz24.svg")),
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        self.index = APP_PAGE.MAIN.value
        self.LPF_Select = Frequency_Selection.LPF_FULL.value
        self.RECORDING = False
        self.MIC_ON = MIC_ON
        self.SOUND_ON = SOUND_ON
        self.selected_camera_index = -1
        self.selected_camera = ''
        self.d435 = None 
        self.adminRole=self.configParams['adminRole'] # Add[if adminRole is True, will can show more features],Brian,05 April 2024
        self.toUseYAML=self.configParams['fourMics']  # true load mic locs from yaml file (for 4 mics case)

        # create the label that holds the image
        self.image_label = Create_ImageWindow()
        # self.image_label.setMinimumSize(QSize(640, 480))
        # self.image_label.setMaximumSize(QSize(1600, 900))
        # self.image_label.setFixedSize(QSize(FRAME_WIDTH, FRAME_HEIGHT))

        self.exit_button = Create_Button("Exit", lambda: self.exit_app(), BUTTON_STYLE_TEXT)
        # self.setting_button = Create_Button("Setting",lambda:switchPage(self,APP_PAGE.SETTING.value),BUTTON_STYLE)
        # self.record_button = Create_Button("Record",self.record_button_clicked,BUTTON_STYLE_RED)

        # Setting Button
        self.setting_button = Create_Button(
            "", lambda: switchPage(self, APP_PAGE.SETTING.value),
            "QPushButton {\n"
            "    color: #333;\n"
            "    border: 0px solid #555;\n"
            "    border-radius: 40px;\n"
            "    border-style: outset;\n"
            "    background: qradialgradient(\n"
            "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
            "        radius: 1.35, stop: 0 #fff, stop: 1 #888\n"
            "        );\n"
            "    padding: 5px;\n"
            "    }\n"
            "\n"
            "QPushButton:hover {\n"
            "    background: qradialgradient(\n"
            "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
            "        radius: 1.35, stop: 0 #fff, stop: 1 #bbb\n"
            "        );\n"
            "    }\n"
            "\n"
            "QPushButton:pressed {\n"
            "    border-style: inset;\n"
            "    background: qradialgradient(\n"
            "        cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,\n"
            "        radius: 1.35, stop: 0 #fff, stop: 1 #ddd\n"
            "        );\n"
            "    }")
        self.setting_button.setText("")
        icon2 = QIcon(resource_path("settings_FILL0_wght400_GRAD0_opsz24.svg"))
        self.setting_button.setIcon(icon2)
        self.setting_button.setIconSize(QSize(65, 65))
        self.setting_button.setObjectName("btnSettings")
        self.setting_button.setFixedSize(NUM_ROUND_BUTTON_SIZE,
                                         NUM_ROUND_BUTTON_SIZE)
        # Record Button
        self.icon_start_record = QIcon(
            resource_path(
                "radio_button_checked_FILL0_wght400_GRAD0_opsz24.png"))
        self.icon_stop_record = QIcon(
            resource_path("stop_circle_FILL0_wght400_GRAD0_opsz24.png"))
        self.record_button = Create_Button(
            "", self.record_button_clicked, "QPushButton {\n"
            "    color: #333;\n"
            "    border: 0px solid #555;\n"
            "    border-radius: 40px;\n"
            "    border-style: outset;\n"
            "    background: qradialgradient(\n"
            "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
            "        radius: 1.35, stop: 0 #fff, stop: 1 #888\n"
            "        );\n"
            "    padding: 5px;\n"
            "    }\n"
            "\n"
            "QPushButton:hover {\n"
            "    background: qradialgradient(\n"
            "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
            "        radius: 1.35, stop: 0 #fff, stop: 1 #bbb\n"
            "        );\n"
            "    }\n"
            "\n"
            "QPushButton:pressed {\n"
            "    border-style: inset;\n"
            "    background: qradialgradient(\n"
            "        cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,\n"
            "        radius: 1.35, stop: 0 #fff, stop: 1 #ddd\n"
            "        );\n"
            "    }")
        self.record_button.setIcon(self.icon_start_record)
        self.record_button.setIconSize(QSize(70, 70))
        self.record_button.setObjectName("btnRecord")
        self.record_button.setFixedSize(NUM_ROUND_BUTTON_SIZE,
                                        NUM_ROUND_BUTTON_SIZE)

        # Mic On Off Button
        self.mic_on_off_button = Create_Button(
            "", self.mic_on_off_button_clicked, "QPushButton {\n"
            "    color: #333;\n"
            "    border: 0px solid #555;\n"
            "    border-radius: 40px;\n"
            "    border-style: outset;\n"
            "    background: qradialgradient(\n"
            "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
            "        radius: 1.35, stop: 0 #fff, stop: 1 #888\n"
            "        );\n"
            "    padding: 5px;\n"
            "    }\n"
            "\n"
            "QPushButton:hover {\n"
            "    background: qradialgradient(\n"
            "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
            "        radius: 1.35, stop: 0 #fff, stop: 1 #bbb\n"
            "        );\n"
            "    }\n"
            "\n"
            "QPushButton:pressed {\n"
            "    border-style: inset;\n"
            "    background: qradialgradient(\n"
            "        cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,\n"
            "        radius: 1.35, stop: 0 #fff, stop: 1 #ddd\n"
            "        );\n"
            "    }")
        self.mic_on_off_button.setText("")
        self.icon_micOn = QIcon(
            resource_path("mic_FILL0_wght400_GRAD0_opsz24.svg"))
        self.icon_micOff = QIcon(
            resource_path("mic_off_FILL0_wght400_GRAD0_opsz24.svg"))
        self.mic_on_off_button.setIconSize(QSize(70, 70))
        self.mic_on_off_button.setObjectName("btnMicOnOff")
        self.mic_on_off_button.setFixedSize(NUM_ROUND_BUTTON_SIZE,
                                            NUM_ROUND_BUTTON_SIZE)
        self.mic_on_off_button.setIcon(self.icon_micOn)

        # Volume On Off Button
        self.sound_on_off_button = Create_Button(
            "", self.sound_on_off_button_clicked, "QPushButton {\n"
            "    color: #333;\n"
            "    border: 0px solid #555;\n"
            "    border-radius: 40px;\n"
            "    border-style: outset;\n"
            "    background: qradialgradient(\n"
            "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
            "        radius: 1.35, stop: 0 #fff, stop: 1 #888\n"
            "        );\n"
            "    padding: 5px;\n"
            "    }\n"
            "\n"
            "QPushButton:hover {\n"
            "    background: qradialgradient(\n"
            "        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
            "        radius: 1.35, stop: 0 #fff, stop: 1 #bbb\n"
            "        );\n"
            "    }\n"
            "\n"
            "QPushButton:pressed {\n"
            "    border-style: inset;\n"
            "    background: qradialgradient(\n"
            "        cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,\n"
            "        radius: 1.35, stop: 0 #fff, stop: 1 #ddd\n"
            "        );\n"
            "    }")
        self.sound_on_off_button.setText("")
        self.icon_soundOn = QIcon(
            resource_path("volume_up_FILL0_wght400_GRAD0_opsz24.svg"))
        self.icon_soundOff = QIcon(
            resource_path("volume_off_FILL0_wght400_GRAD0_opsz24.svg"))
        self.sound_on_off_button.setIcon(self.icon_soundOn)
        self.sound_on_off_button.setIconSize(QSize(70, 70))
        self.sound_on_off_button.setObjectName("btnSoundOnOff")
        self.sound_on_off_button.setFixedSize(NUM_ROUND_BUTTON_SIZE,
                                              NUM_ROUND_BUTTON_SIZE)

        self.slider_mic_vol = QSlider()
        self.slider_mic_vol.setOrientation(Qt.Horizontal)
        self.slider_mic_vol.setObjectName("slider_mic_vol")
        self.slider_mic_vol.setFixedSize(200, 40)
        self.slider_mic_vol.setStyleSheet(SLIDER_STYLE_2)

        self.slider_sound_vol = QSlider()
        self.slider_sound_vol.setOrientation(Qt.Horizontal)
        self.slider_sound_vol.setObjectName("slider_sound_vol")
        self.slider_sound_vol.setMinimum(0)
        self.slider_sound_vol.setMaximum(100)
        self.slider_sound_vol.setFixedSize(200, 40)
        self.slider_sound_vol.setStyleSheet(SLIDER_STYLE_2)

        self.text_label = Create_Label()

        #Setting Up Main Page
        self.main_page = QGridLayout()
        self.main_page.setContentsMargins(0, 0, 0, 0)
        self.main_page.setHorizontalSpacing(
            0)  # Set horizontal spacing to zero
        self.main_page.addWidget(self.image_label,
                                 0,
                                 0,
                                 1,
                                 2,
                                 alignment=Qt.AlignCenter)
        self.main_page.setHorizontalSpacing(
            0)  # Set horizontal spacing to zero
        self.main_page.setVerticalSpacing(0)  # Set horizontal spacing to zero
        
        

        self.main_page_button = QGridLayout()
        self.main_page_button.setContentsMargins(100, 0, 100, 50)
        self.main_page_button.addWidget(self.exit_button, 0, 0, Qt.AlignLeft)
        self.main_page_button.addWidget(self.record_button, 0, 1, 1, 2,
                                        Qt.AlignCenter)
        self.button_slider_layout = QGridLayout()
        self.button_slider_layout.addWidget(self.slider_mic_vol, 0, 0, 1, 1,
                                            Qt.AlignCenter)
        self.button_slider_layout.addWidget(self.mic_on_off_button, 1, 0, 1, 1,
                                            Qt.AlignCenter)
        self.button_slider_layout.addWidget(self.slider_sound_vol, 0, 1, 1, 1,
                                            Qt.AlignCenter)
        self.button_slider_layout.addWidget(self.sound_on_off_button, 1, 1, 1,
                                            1, Qt.AlignCenter)
        self.button_slider_layout.addWidget(self.setting_button, 0, 3, 2, 1,
                                            Qt.AlignCenter)
        
        self.button_slider_layout.setSpacing(20)
        self.main_page_button.addLayout(self.button_slider_layout, 0, 2, 1, 2,
                                        Qt.AlignRight)
        

        
        self.main_page_button_widget = QWidget()
        self.main_page_button_widget.setLayout(self.main_page_button)
        self.main_page_button_widget.setFixedSize(WINDOW_WIDTH,
                                                  BUTTON_BAR_HEIGHT)
        # self.main_page_button_widget.setStyleSheet("background-color:transparent; border:1px solid rgb(0, 255, 0);")

        # removed the text_label for debugging
        # if DEBUG == True:
        #     self.main_page.addWidget(self.text_label,
        #                              0,
        #                              0,
        #                              1,
        #                              2,
        #                              alignment=Qt.AlignRight)
        #     self.main_page.addWidget(self.main_page_button_widget, 1, 0, 1, 2)
        # else:
        #     self.main_page.addWidget(self.main_page_button_widget, 1, 0, 1, 2,
        #                              Qt.AlignCenter)
        self.main_page.addWidget(self.main_page_button_widget, 1, 0, 1, 2,Qt.AlignCenter)

        # setup the setting page 
        self.setupSettingsPage()
        

        self.main_page_widget = QWidget()
        self.main_page_widget.setLayout(self.main_page)
        self.main_page_widget.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)


        self.test_page_widget = self.setupTestPageUI()
        self.test_page_widget.setFixedSize(WINDOW_WIDTH,WINDOW_HEIGHT)
        
        

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.stacked_widget.setContentsMargins(0, 0, 0, 0)
        self.stacked_widget.addWidget(self.main_page_widget)
        self.stacked_widget.addWidget(self.setting_page_widget)
        self.stacked_widget.addWidget(self.test_page_widget)
        self.stacked_widget.setCurrentIndex(0) 

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.stacked_widget)
        # revised[zero margins],Brian, 1 April 2024
        #self.layout.setContentsMargins(0, 50, 0, 0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.stacked_widget)
        # set the vbox layout as the widgets layout
        self.setLayout(self.layout)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        # self.setGeometry(QApplication.screens()[self.screen_number].geometry())
        # self.showFullScreen()

        # pop up camera_dialog to select video first
        camera_dialog = CameraSelectionDialog(get_camera_list())
        # Display the camera_dialog and get the selected camera name
        if camera_dialog.exec_() == QDialog.Accepted:
            self.selected_camera_index, self.selected_camera = camera_dialog.get_selected_camera_name()


            audio_inputDevices = MyAudioUtilities.GetAudioDevices(direction='in')
            audio_outputDevices= MyAudioUtilities.GetAudioDevices(direction='out')

            input_devices = [
                '%d,%s' % (i, MyAudioUtilities.dev_to_str(device))
                for i, device in enumerate(audio_inputDevices)
            ]

            output_devices = [
                '%d,%s' % (i, MyAudioUtilities.dev_to_str(device))
                for i, device in enumerate(audio_outputDevices)
            ]
            
            audio_dialog = AudioDeviceDialog(input_devices, output_devices,self.windowIcon())

            
            # Display the audio_dialog and get the selected camera name
            if audio_dialog.exec_() == QDialog.Accepted:
                self.input_device, self.output_device = audio_dialog.get_selected_devices()

                # retrieve device uuids
                index = int(self.input_device.split(',')[0])
                self.input_devid = audio_inputDevices[index].id
                self.input_device= audio_inputDevices[index].FriendlyName

                index = int(self.output_device.split(',')[0])
                self.output_devid= audio_outputDevices[index].id

                devices = sd.query_devices()
                input_devices = [
                    '%d,%s' % (i, audio_dev_to_str(device))
                    for i, device in enumerate(devices)
                    if device['name'] == self.input_device
                ]

                if len(input_devices)>0:
                    # select the first one if we have more than one input devices with the same name
                    self.input_device = int(input_devices[0].split(',')[0])
                else:
                    # failed to locate input audio device using sd and have to quit now
                    self.showAlert('Error','Failed to select the right audio input device! Have to exit now!',self.windowIcon())
                    exit()

                print("input device: ", self.input_device,self.input_devid)
                print("output device: ", self.output_device,self.output_devid)

                # create audio controller for the output device
                self.audio_outCtrl = AudioController(self.output_devid,'output','default')
                self.audio_outCtrl.listAllSections()
                print(self.audio_outCtrl.volume)

                # turn to zero volume at start
                self.audio_outCtrl.set_volume(0)
                self.slider_sound_vol.setValue(0)
                self.slider_sound_vol.valueChanged.connect(self.audio_outCtrl.set_volume)

                # create audio controller for the input device
                self.audio_inCtrl = AudioController(self.input_devid,'input','default')

                # turn to zero volume at start
                self.audio_inCtrl.set_volume(0)
                self.slider_mic_vol.setValue(0)
                self.slider_mic_vol.valueChanged.connect(self.audio_inCtrl.set_volume)

                # create the video capture thread
                self.video_thread = VideoThread(self.selected_camera_index,self.selected_camera, App.CAM_DISPLAY_WIDTH, App.CAM_DISPLAY_HEIGHT, self.configParams['usePoseEstimation'])
                self.audio_thread = AudioThread(self.input_device)

                # send packet when depth locked, Jason, 16 April 2024
                if hasattr(self.video_thread, 'd435'):
                    self.video_thread.depth_value_locked.connect(self.send_3d_point)

                # connect its signal to the update_image slot
                self.video_thread.change_pixmap_signal.connect(self.update_image)
                # connnect signal to update 3d coordinate
                self.video_thread.update_3d_coordinate.connect(self.update_3d_coordinate)
                # start video thread
                if self.video_thread.d435:
                    self.video_thread.d435.start()
                elif self.video_thread.cam:
                    self.video_thread.cam.start()
                self.video_thread.start()

                self.logger = CustomProgressBarLogger()
            else:
                raise Exception('exit from init')

            
            
        else:
            # print("Exit now...")
            raise Exception('exit from init')

        
    def tryLoadConfig(self):
        '''
        check if config.yaml exists, if yes, load config params from it, else set default values manually

        '''
        configFileName = 'config.yaml'
        # configFilePath = config_path(configFileName)
        print('Config Path: ', configFileName)
        if os.path.exists(configFileName):
            with open(configFileName, 'r') as file:
                configParams= yaml.safe_load(file)
                return configParams
        else:
            # set up default values manually
            self.configParams = {
                'debug': False,
                'fourMics': True,
                'adminRole': False,
                'showROI': True,
                'usePoseEstimation': False,
            }
            return self.configParams

        
        
    def showAdminWidgets(self):
        self.adminFrame.show()

    def showPasswordDialog(self):
        '''
        pop up password dialog to try to enter admin mode
        '''

        password, ok = QInputDialog.getText(self,"Enter Password", "Password:", QLineEdit.Password)
        
        if ok and password == "1729":
            self.adminRole=True
            self.showAdminWidgets()
            QMessageBox.information(self, "Admin Mode", "Admin Mode Entered")
            
            # Perform actions for test mode
        else:
            self.adminRole=False
            QMessageBox.warning(self, "Incorrect Password", "The password is incorrect.")


    def cleanUp(self):
        if hasattr(self, 'mouse_press_timer'):
            self.mouse_press_timer.stop()
        
    def send_message(self, message):
        pass

    def toggleDebugMode(self):
        global DEBUG
        if DEBUG:
            DEBUG=False
            self.btnToggleDebug.setText('Turn On Debug')
        else:
            DEBUG=True
            self.btnToggleDebug.setText('Turn Off Debug')

    # Jason - 8 April 2024 - Aligned Frames
    def toggleAlignedFramesMode(self):
        global ALIGNED_FRAMES, FILTERED_FRAMES
        ALIGNED_FRAMES = not ALIGNED_FRAMES
        if ALIGNED_FRAMES:
            self.btnToggleAlignedFrames.setText('Turn Off Aligned Frame')
        else:
            FILTERED_FRAMES = False
            self.btnToggleAlignedFrames.setText('Turn On Aligned Frame')  
        print("ALIGNED_FRAMES: ", ALIGNED_FRAMES)

    # Added, Jason - 12 April 2024
    def toggleFilteredFramesMode(self):
        global ALIGNED_FRAMES, FILTERED_FRAMES
        if ALIGNED_FRAMES:
            FILTERED_FRAMES = not FILTERED_FRAMES
            if FILTERED_FRAMES:
                self.btnToggleFilteredFrames.setText('Turn Off Filtered Frame')
            else:
                self.btnToggleFilteredFrames.setText('Turn On Filtered Frame')
        print("FILTERED_FRAMES: ", FILTERED_FRAMES)

    def exitAdminMode(self):
        '''
        reset adminRole DEUGB and hide adminFrame
        '''
        global DEBUG
        self.adminRole=False
        DEBUG=False
        self.adminFrame.hide()

    def goTestMode(self):
        '''
        go to stacked widget index 2
        '''
        self.stacked_widget.setCurrentIndex(2)

    
    def setTargetPos(self,point):
        
        # get handle of tbx_targetPos
        theWidget = [textbox for textbox in self.test_page_widget.findChildren(QLineEdit) if textbox.objectName()=='tbx_targetPos']
        if len(theWidget)>0:
            # remember that x, y need to *-1 !!
            theWidget[0].setText('%.2f,%.2f,%.2f' %(-point[0],-point[1],point[2]))    
        

    def setupSettingsPage(self):
        # revised[using Clickable Label instead],Brian, 05 April 2024
        # Setting up Setting page for LPF and Gain and Voulme
        # self.gain_label = QLabel("Mic Array Channel Gain  :")
        self.gain_label = ClickableLabel("Mic Array Channel Gain  :")
        # button_font = QFont("Arial",40)
        # button_font.setPixelSize(40)
        self.gain_label.setFont(BUTTON_FONT)
        self.gain_label.clicked.connect(self.showPasswordDialog)

        self.volume_label = QLabel("Mic Array Digital Volume :")
        self.volume_label.setFont(BUTTON_FONT)
        self.gain_fader = Create_Slider(-12, 12, 0, 1, SLIDER_STYLE_2,
                                        function=update_label)
        self.volume_fader = Create_Slider(0, 24, 0, 1, SLIDER_STYLE_2,
                                          function=update_label)
        self.filter_select_label = QLabel("Mic Array Filter Select     :")
        self.filter_select_label.setFont(BUTTON_FONT)

        self.checkbox_6kHz = Create_RadioBotton(
            '6khz',
            lambda: ToggleSelection(self, Frequency_Selection.LPF_6K.value))
        self.checkbox_12kHz = Create_RadioBotton(
            '12khz',
            lambda: ToggleSelection(self, Frequency_Selection.LPF_12K.value))
        self.checkbox_18kHz = Create_RadioBotton(
            '18khz',
            lambda: ToggleSelection(self, Frequency_Selection.LPF_18K.value))
        self.checkbox_full = Create_RadioBotton(
            'Full Range',
            lambda: ToggleSelection(self, Frequency_Selection.LPF_FULL.value))
        self.back_button = Create_Button(
            "Back", lambda: switchPage(self, APP_PAGE.MAIN.value),
            BUTTON_STYLE_TEXT)
        self.ApplyButton = Create_Button("Apply", lambda: exit(), BUTTON_STYLE_TEXT)

        self.setting_page = QGridLayout()
        self.setting_page.addWidget(self.gain_label, 1, 0, 1, 1)
        self.setting_page.addWidget(self.gain_fader, 1, 1, 1, 4,
                                    Qt.AlignHCenter)

        self.setting_page.addWidget(self.volume_label, 2, 0, 1, 1)
        self.setting_page.addWidget(self.volume_fader, 2, 1, 1, 4,
                                    Qt.AlignHCenter)

        self.setting_page.addWidget(self.filter_select_label, 3, 0, 1, 1)
        self.setting_page.addWidget(self.checkbox_6kHz, 3, 1, 1, 1,
                                    Qt.AlignHCenter)
        self.setting_page.addWidget(self.checkbox_12kHz, 3, 2, 1, 1,
                                    Qt.AlignHCenter)
        self.setting_page.addWidget(self.checkbox_18kHz, 3, 3, 1, 1,
                                    Qt.AlignHCenter)
        self.setting_page.addWidget(self.checkbox_full, 3, 4, 1, 1,
                                    Qt.AlignHCenter)

        self.setting_page.addWidget(self.ApplyButton, 4, 3, 1, 1,
                                    Qt.AlignHCenter)
        self.setting_page.addWidget(self.back_button, 4, 4, 1, 1,
                                    Qt.AlignHCenter)
        
        # show the adminFrame is 'adminRole' is True
        self.adminFrame = self.setupAdminFrame(self.configParams['adminRole'])
        self.setting_page.addWidget(self.adminFrame,5,0,1,1)
        


        self.setting_page_widget = QWidget()
        self.setting_page_widget.setLayout(self.setting_page)
        self.setting_page_widget.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        # add[change background to gray],Brian,05 April 2024
        self.setting_page_widget.setStyleSheet("background-color:gray")

    def setupAdminFrame(self,toShow=False):
        '''
        create Frame for Admin Mode Features

        '''
        adminLayout= QGridLayout()

        lbl_AdminMode = QLabel('Admin Mode:')
        lbl_AdminMode.setStyleSheet(LABEL_STYLE_ADMIN_FRAME)
        adminLayout.addWidget(lbl_AdminMode,0,0,1,1)

        self.lbl_info = QLabel('Info:')
        adminLayout.addWidget(self.lbl_info)
        
        btnTestMode= QPushButton('Go to Test mode Page')
        btnTestMode.setStyleSheet(BUTTON_STYLE_ADMIN_FRAME)
        btnTestMode.clicked.connect(self.goTestMode)


        self.btnToggleDebug= QPushButton('Turn On Debug')
        self.btnToggleDebug.setStyleSheet(BUTTON_STYLE_ADMIN_FRAME)
        self.btnToggleDebug.clicked.connect(self.toggleDebugMode)

        self.btnToggleAlignedFrames = QPushButton('Turn On Aligned Frames')
        self.btnToggleAlignedFrames.setStyleSheet(BUTTON_STYLE_ADMIN_FRAME)
        self.btnToggleAlignedFrames.clicked.connect(self.toggleAlignedFramesMode)

        self.btnToggleFilteredFrames = QPushButton('Turn On Filtered Mode')
        self.btnToggleFilteredFrames.setStyleSheet(BUTTON_STYLE_ADMIN_FRAME)
        self.btnToggleFilteredFrames.clicked.connect(self.toggleFilteredFramesMode)

        btnExitAdminMode= QPushButton('Exit Admin Mode')
        btnExitAdminMode.setStyleSheet(BUTTON_STYLE_ADMIN_FRAME)
        btnExitAdminMode.clicked.connect(self.exitAdminMode)

        adminLayout.addWidget(btnTestMode,1,0,1,1)
        adminLayout.addWidget(self.btnToggleDebug,1,1,1,1)
        adminLayout.addWidget(self.btnToggleAlignedFrames,2,0,1,1)
        adminLayout.addWidget(self.btnToggleFilteredFrames,2,1,1,1)
        adminLayout.addWidget(btnExitAdminMode,3,0,1,1)

        adminFrame = QFrame()
        adminFrame.setFrameStyle(QFrame.Panel | QFrame.Plain)
        adminFrame.setStyleSheet("#innerFrame { border: 1px solid blue; }")
        adminFrame.setFixedWidth(1500)
        adminFrame.setFixedHeight(300)
        adminFrame.setLayout(adminLayout)
        if toShow:
            adminFrame.show()
        else:
            adminFrame.hide()
        return adminFrame
        
        
    def setupTestPageUI(self):
        '''
        to create ui for the test page

        retuns a QWidget
        '''

        modes = [
            '0: normal', 
            '1: cal',
            '2: cal verify',
            '3: switch mic/output selection',
            '4: turn on BM',
            '5: turn off BM',
            '6: turn on MC',
            '7: turn off MC',
            '8: BLjudge H_CAFFs readback',
            '9: WMcal Wm[] readback',
            
        ]
        
        self.micNames=["M{:02d}".format(i) for i in range(1, 33)]

        setTests=[
            '0: PS_enMC-0,PS_enBM-0,FFTgain: 2, MIC8-data_bm_n, MIC9-ifftout',
            '4: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n, MIC9-ifftout',
            '8: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n, MIC9-ifftout, delta_t=0',
            '9: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n, MIC9-data_fbf_d_MC',
            '10: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_ym_n, MIC9-data_fbf_d_MC, delta_t=0',
            '11: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n_dly, MIC9-data_bm_0',
            '12: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n_dly, MIC9-data_bm_0, delta_t=0',
            '13: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_0, MIC9-data_ym_0',
            '14: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_ym_judge, MIC9-data_bm_0'
        ]

        name_dict = {
            'lbl_hostIP': {'text':'Host IP:','row':1,'column':0,'row_span':1,'col_span':1},
            'tbx_hostIP': {'text':'192.168.1.40','row':1,'column':1,'row_span':1,'col_span':1},
            'lbl_hostPort':{'text':'Host Port','row':1,'column':2,'row_span':1,'col_span':1},
            'tbx_hostPort':{'text':'5004','row':1,'column':3,'row_span':1,'col_span':1},
            'lbl_mode': {'text':'mode','row':2,'column':0,'row_span':1,'col_span':1},
            'cbx_mode': {'items':modes,'row':2,'column':1,'row_span':1,'col_span':1},
            'lbl_micNum': {'text':'mic#','row':2,'column':2,'row_span':1,'col_span':1},
            'cbx_micNum': {'items':self.micNames,'row':2,'column':3,'row_span':1,'col_span':1},
            'lbl_micGain': {'text':'mic gain','row':3,'column':0,'row_span':1,'col_span':1},
            'tbx_micGain': {'text':'30','row':3,'column':1,'row_span':1,'col_span':1},
            'lbl_micDisable':{'text':'mic disable','row':3,'column':2,'row_span':1,'col_span':1},
            'tbx_micDisable':{'text':'30','row':3,'column':3,'row_span':1,'col_span':1},
            'lbl_setTest': {'text':'set Test','row':4,'column':0,'row_span':1,'col_span':1},
            'cbx_setTest': {'items':setTests,'row':4,'column':1,'row_span':1,'col_span':1},
            'lbl_denOutSel':{'text':'den_out_sel','row':5,'column':0,'row_span':1,'col_span':1},
            'tbx_denOutSel':{'text':'8','row':5,'column':1,'row_span':1,'col_span':1},
            'lbl_mcBetaSel':{'text':'mc_beta_sel','row':5,'column':2,'row_span':1,'col_span':1},
            'tbx_mcBetaSel':{'text':'4','row':5,'column':3,'row_span':1,'col_span':1},
            'lbl_mcKSel':{'text':'mc_K_sel','row':6,'column':0,'row_span':1,'col_span':1},
            'tbx_mcKSel':{'text':'1','row':6,'column':1,'row_span':1,'col_span':1},
            'lbl_en_BM_MC_ctrl':{'text':'en_BM_MC_ctrl','row':6,'column':2,'row_span':1,'col_span':1},
            'tbx_en_BM_MC_ctrl':{'text':'3','row':6,'column':3,'row_span':1,'col_span':1},
            'lbl_targetPos':{'text':'target pos','row':9,'column':0,'row_span':1,'col_span':1},
            'tbx_targetPos':{'text':'0,0,0','row':9,'column':1,'row_span':1,'col_span':1},
            'lbl_xyzOffsets':{'text':'x,y,z Offsets','row':10,'column':0,'row_span':1,'col_span':1},
            'tbx_xyzOffsets':{'text':'0,0,0','row':10,'column':1,'row_span':1,'col_span':1},
            'lbl_fourMics':{'text':'4 Mics','row':10,'column':3,'row_span':1,'col_span':1},
            'tbx_fourMics':{'text':'1' if self.toUseYAML else '0','row':10,'column':4,'row_span':1,'col_span':1},
        }

        widget = QWidget()
        outer_layout = QVBoxLayout()

        layout = QGridLayout()
        lbl_title = QLabel('Test Page')
        lbl_title.setStyleSheet(LABEL_STYLE_TEST_PAGE)
        layout.addWidget(lbl_title,0,0,1,1)

        points = [(50, 50), (100, 100), (150, 150), (200, 200)]
        #point_selection = PointSelectionGUI(points,self.send_message)
        # echoText = EchoText() 
        # layout.addWidget(echoText,16,5,1,1)
        
        # Create and add QLabel and QLineEdit widgets dynamically
        for name, properties in name_dict.items():
            if name.startswith("lbl_"):
                label = QLabel()
                label.setText(properties["text"])
                label.setObjectName(name)
                label.setStyleSheet(LABEL_STYLE_TEST_PAGE)
                layout.addWidget(
                    label,
                    properties["row"],
                    properties["column"],
                    properties["row_span"],
                    properties["col_span"],
                )
            elif name.startswith("tbx_"):
                line_edit = QLineEdit()
                line_edit.setText(properties['text'])
                line_edit.setObjectName(name)
                line_edit.setStyleSheet(LINEEDIT_STYLE_TEST_PAGE)
                layout.addWidget(
                    line_edit,
                    properties["row"],
                    properties["column"],
                    properties["row_span"],
                    properties["col_span"],
                )
            elif name.startswith("cbx_"):
                combo_box = QComboBox()
                combo_box.setObjectName(name)
                combo_box.setStyleSheet(COMBO_STYLE_TEST_PAGE)
                combo_box.addItems(properties["items"])  # Add items to the combobox
                layout.addWidget(
                    combo_box,
                    properties["row"],
                    properties["column"],
                    properties["row_span"],
                    properties["col_span"],
                )

        # Revised to be a start button to enable sending packet after mouse click,
        # Jason, 17 April 2024
        btnSendPacket = QPushButton('Send Packet')
        btnSendPacket.setStyleSheet(BUTTON_STYLE_TEST_PAGE)
        btnSendPacket.clicked.connect(self.sendPacket)
        # btnSendPacket.clicked.connect(self.onSendPacketButtonClicked)

        btnTestPing = QPushButton('Test Ping Host')
        btnTestPing.setStyleSheet(BUTTON_STYLE_TEST_PAGE)
        btnTestPing.clicked.connect(self.testPingHost)

        btnExitTestPage = QPushButton('Exit Test Page')
        btnExitTestPage.setStyleSheet(BUTTON_STYLE_TEST_PAGE)
        btnExitTestPage.clicked.connect(self.exitTestPage)


        self.lbl_msg = QLabel('')
        self.lbl_msg.setStyleSheet(SMALL_LABEL_STYLE_TEST_PAGE)

        layout.addWidget(btnSendPacket,14,0,1,1)
        layout.addWidget(btnTestPing,15,0,1,1)
        layout.addWidget(btnExitTestPage,16,0,1,1)
        layout.addWidget(self.lbl_msg,17,0,1,3)
        
        # Set border color and width for the inner layout
        frame = QFrame()
        frame.setObjectName("innerFrame")
        frame.setFrameStyle(QFrame.Panel | QFrame.Plain)
        #frame.setStyleSheet("#innerFrame { border: 1px solid blue; }")
        frame.setLineWidth(1)
        frame.setMidLineWidth(1)
        frame.setLayout(layout)
        frame.setFixedHeight(1000)
        frame.setFixedWidth(1500)
        # frame.setMinimumSize(400,300)
        # frame.setMaximumSize(1000,600)



        outer_layout.addWidget(frame,alignment=Qt.AlignTop)
        widget.setLayout(outer_layout)

        


        return widget
    

    def __str__(self):
        return f"params,{self.hostIP}, {self.hostPort}, {self.mode},{self.micIndx},{self.micGain},{self.setTest},{self.den_out_sel},{self.mc_beta_sel},{self.mc_K_sel},{self.en_BM_MC_ctrl},[{self.targetPos[0]},{self.targetPos[1]},{self.targetPos[2]}],[{self.offsets[0]},{self.offsets[1]},{self.offsets[2]}],{self.toUseYAML}"


    def printParams(self):
        global dataLogger
        print(self.hostIP,self.hostPort,self.mode,self.micIndx,self.micGain,self.setTest,self.den_out_sel,self.mc_beta_sel,self.mc_K_sel,self.targetPos,self.offsets,self.toUseYAML)
        # add[save to log as well],Brian,27 Mar 2024
        dataLogger.add_data(self.__str__())

    def fetchParamsFromUI(self):

        # name_dict = {
        #     'cbx_mode': {'items':modes,'row':3,'column':1,'row_span':1,'col_span':1},
        #     'cbx_micNum': {'items':micNames,'row':4,'column':1,'row_span':1,'col_span':1},
        #     'cbx_setText': {'items':setTests,'row':7,'column':1,'row_span':1,'col_span':1},


        
        
        # get values from textboxes
        textboxes = self.test_page_widget.findChildren(QLineEdit)
        params = {textbox.objectName(): textbox.text() for textbox in textboxes}
        
        
        self.hostIP     = params['tbx_hostIP']
        self.hostPort   = int(params['tbx_hostPort'])
        self.micGain    = int(params['tbx_micGain'])
        self.micDisable = int(params['tbx_micDisable'])
        self.den_out_sel   = int(params['tbx_denOutSel'])
        self.mc_beta_sel   = int(params['tbx_mcBetaSel'])
        self.mc_K_sel      = int(params['tbx_mcKSel'])
        self.en_BM_MC_ctrl = int(params['tbx_en_BM_MC_ctrl'])
        self.targetPos     = np.array(params['tbx_targetPos'].split(','), dtype=float)
        self.offsets       = np.array(params['tbx_xyzOffsets'].split(','), dtype=float)
        self.toUseYAML     = True if params['tbx_fourMics'] == '1' else False

        # get values from combo boxes
        comboboxes = self.test_page_widget.findChildren(QComboBox)
        params = {cbx.objectName(): cbx.currentText() for cbx in comboboxes}
        
        self.micIndx    = self.micNames.index(params['cbx_micNum'])
        self.mode       = int(params['cbx_mode'].split(':')[0])
        self.setTest    = int(params['cbx_setTest'].split(':')[0])


    def showInfo(self,sMsg):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lbl_msg.setText('%s, %s' %(timestamp,sMsg))
    
    # Revised to be a start button to enable sending packet after mouse click,
    # Jason, 17 April 2024
    # def onSendPacketButtonClicked(self):
    #     global START_SEND_PACKET
    #     START_SEND_PACKET = not START_SEND_PACKET
    #     if START_SEND_PACKET:
    #         self.btnSendPacket.setText('Stop Send Packet')
    #     else:
    #         self.btnSendPacket.setText('Send Packet')

    def sendPacket(self):
        global dataLogger

        # clear lbl_info first
        self.showInfo('')
        sendBuf=b'SET0'
        
        self.setTargetPos(self.targetPos)

        self.fetchParamsFromUI()
        self.printParams()
    
        #Z=distance between camera and object, x is left+/right-, y is down+/up-
        
        # just a dummy target location
        this_location=[6, 0.2, 0.3]

        # revised[add offsets],Brian,18 Mar 2024
        delay=delay_calculation_v1(this_location)
        print(delay)
        
        #converting the delay into binary format 
        delay_binary_output = delay_to_binary(delay)
        #print(delay_binary_output)
        #need to do later
        RW_field=[1,1]
        mode=0
        mic_gain=[1,0]
        mic_num=0
        en_bm=1
        en_bc=1
        mic_en=1
        type=0
        reserved=0
        message=struct_packet(RW_field,mode,mic_gain,mic_num,en_bm,en_bc,delay_binary_output[0],mic_en,type,reserved)
        print(message)
        messagehex = BintoINT(message)
        print(messagehex)
        message1 = int(messagehex[2:4],16) # hex at  1 and 2  
        message2 = int(messagehex[4:6],16) # hex at  3 and 4 
        message3 = int(messagehex[6:8],16)  # hex at  5 and 6 
        message4 = int(messagehex[8:],16)
        print("m1:{},m2:{},m3:{},m4:{}\n".format(message1,message2,message3,message4))
    

        message5  = int(self.mode)        # mode
        message6  = int(self.micIndx)     # mic
        message7  = int(self.micGain)     # mic_gain
        message8  = int(self.micDisable)  # mic_disable
        message9  = int(self.setTest)     # set_test
        message10 = int(self.den_out_sel) # den_out_sel, previously micDelay

        # revise[added message11, message12],Brian, 27 Mar 2024
        message11 = int(self.mc_beta_sel) # mc_beta_sel
        message12 = int(self.mc_K_sel)    # mc_K_sel

        # revise[added message13],Brian, 28 Mar 2024
        message13 = int(self.en_BM_MC_ctrl) # en_BM_MC_ctrl
 
        
        _,refDelay,_ = delay_calculation(self.targetPos,self.offsets[0],self.offsets[1],self.offsets[2],toUseYAML=self.toUseYAML)   
        
        # save a copy of the raw delay in us
        rawDelay = refDelay[1:]*1e6
        # revise[should not include m00],Brian,15 April 204
        refDelay = refDelay[1:]
        refDelay = refDelay*48e3
        refDelay = np.max(refDelay)-refDelay
        refDelay = np.round(refDelay)

        #convert refDelay to byte
        #but make sure that they are within 0 to 255 first!!
        assert (refDelay>=0).all() and (refDelay<=255).all()

            
        refDelay = refDelay.astype(np.uint8)
        payload = refDelay.tobytes()
        print('refDelay',refDelay)
        print('payload',payload)
        print('sendBuf',sendBuf)

        

        
        packet = prepareMicDelaysPacket(payload)
        if validateMicDelaysPacket(packet):
            print('packet ok')
        else:
            print('packet not ok')
            
        sendBuf=bytes([message1,message2,message3,message4,message5,message6,message7,message8,message9,message10,message11,message12,message13])
            
        # append packet to sendBuf
        sendBuf += packet
        dataLogger.add_data('data,%s,%s,%s,%s' %(bytes(sendBuf),np.array2string(refDelay),np.array2string(np.array(self.targetPos)),np.array2string(rawDelay)))


        if send_and_receive_packet(self.hostIP,self.hostPort,sendBuf,timeout=3):
            print('data transmission ok')
            self.showInfo('tx ok')
            dataLogger.add_data('tx ok')
        else:
            print('data transmission failed')
            dataLogger.add_data('tx failed')



    def sendPacket_v2(self):
        global dataLogger

        # clear lbl_info first
        self.showInfo('')
        sendBuf=b'SET0'
        
        
        #Z=distance between camera and object, x is left+/right-, y is down+/up-
        
        # just a dummy target location
        this_location=[6, 0.2, 0.3]

        # revised[add offsets],Brian,18 Mar 2024
        delay=delay_calculation_v1(this_location)
        print(delay)
        
        #converting the delay into binary format 
        delay_binary_output = delay_to_binary(delay)
        #print(delay_binary_output)
        #need to do later
        RW_field=[1,1]
        mode=0
        mic_gain=[1,0]
        mic_num=0
        en_bm=1
        en_bc=1
        mic_en=1
        type=0
        reserved=0
        message=struct_packet(RW_field,mode,mic_gain,mic_num,en_bm,en_bc,delay_binary_output[0],mic_en,type,reserved)
        print(message)
        messagehex = BintoINT(message)
        print(messagehex)
        message1 = int(messagehex[2:4],16) # hex at  1 and 2  
        message2 = int(messagehex[4:6],16) # hex at  3 and 4 
        message3 = int(messagehex[6:8],16)  # hex at  5 and 6 
        message4 = int(messagehex[8:],16)
        print("m1:{},m2:{},m3:{},m4:{}\n".format(message1,message2,message3,message4))
    

        message5  = int('0')        # mode
        message6  = int('1')        # mic
        message7  = int('10')     # mic_gain
        message8  = int('30')  # mic_disable
        message9  = int('9')     # set_test
        message10 = int('4') # den_out_sel, previously micDelay

        # revise[added message11, message12],Brian, 27 Mar 2024
        message11 = int('1') # mc_beta_sel
        message12 = int('3')    # mc_K_sel

        # revise[added message13],Brian, 28 Mar 2024
        message13 = int('3') # en_BM_MC_ctrl
 
        
        _,refDelay,_ = delay_calculation(self.targetPos,0,-0.1,0)   
        # revise[should not include m00],Brian,15 April 204
        refDelay = refDelay[1:]
        refDelay = refDelay*48e3
        refDelay = np.max(refDelay)-refDelay
        refDelay = np.round(refDelay)

        #convert refDelay to byte
        #but make sure that they are within 0 to 255 first!!
        assert (refDelay>=0).all() and (refDelay<=255).all()

            
        refDelay = refDelay.astype(np.uint8)
        payload = refDelay.tobytes()
        print('refDelay',refDelay)
        print('payload',payload)
        print('sendBuf',sendBuf)

        

        
        packet = prepareMicDelaysPacket(payload)
        if validateMicDelaysPacket(packet):
            print('packet ok')
        else:
            print('packet not ok')
            
        sendBuf=bytes([message1,message2,message3,message4,message5,message6,message7,message8,message9,message10,message11,message12,message13])
            
        # append packet to sendBuf
        sendBuf += packet
        dataLogger.add_data('data,%s,%s,%s' %(bytes(sendBuf),np.array2string(refDelay),np.array2string(np.array(self.targetPos))))


        if send_and_receive_packet(self.hostIP,self.hostPort,sendBuf,timeout=3):
            print('data transmission ok')
            self.showInfo('tx ok')
            dataLogger.add_data('tx ok')
        else:
            print('data transmission failed')
            dataLogger.add_data('tx failed')



    def updatePingResults(self,results):
        if results[0]:
            self.lbl_msg.setText('host is alive')
        else:
            self.lbl_msg.setText('host not available!!')

        self.thread.quit()
        self.thread.wait()
    
    def testPingHost(self):
        '''
        try to ping host to see if it's alive

        '''
        self.lbl_msg.setText('trying to ping host...')
        textboxes = self.test_page_widget.findChildren(QLineEdit)
        hostIPs = [textbox.text() for textbox in textboxes if textbox.objectName()=='tbx_hostIP']
        if len(hostIPs)>0:
            self.thread = QThread()
            self.worker = WorkerTryPing()
            self.worker.setHostIP(hostIPs[0])
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.updatePingResults)
            self.thread.start()
        


    def exitTestPage(self):
        '''
        exit test page, back to settings page
        '''
        self.stacked_widget.setCurrentIndex(1)
        
        




        
        
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @pyqtSlot(list)
    def update_3d_coordinate(self,point):
        '''
        update 3d coodrinate
        '''

        self.targetPos = point
        

    # revised,Brian, 1 April 2024
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(WINDOW_WIDTH, WINDOW_HEIGHT, Qt.KeepAspectRatio)
        # p = convert_to_Qt_format.scaled(FRAME_WIDTH, FRAME_HEIGHT, Qt.KeepAspectRatio)
        # p = convert_to_Qt_format.scaled(FRAME_WIDTH, FRAME_HEIGHT, Qt.IgnoreAspectRatio)
        return QPixmap.fromImage(p)

    def mousePressEvent(self, event):
        global MOUSE_CLICKED, START_SEND_PACKET
        # Handle mouse press events
        mouse_position = event.pos()
        # try to get 3d coordinate from camera if it's d435
        # based on mouse_position.x(), y()

        # revised[make sure that stacked_widget is at index 0],Brian,05 April 2024
        if self.stacked_widget.currentIndex()==0:
            if self.selected_camera.startswith('Intel(R) RealSense(TM) Depth Camera 4') and self.selected_camera.endswith('RGB'):
                currentX = mouse_position.x()
                currentY = mouse_position.y()

                self.video_thread.setMouseXY(currentX,currentY)

                # if self.adminRole:
                #     self.setTargetPos(self.targetPos)

            self.text_label.appendPlainText(
                f"Clicked on [{mouse_position.x()},{mouse_position.y()}]")
            row = int(mouse_position.y()) // (WINDOW_HEIGHT // 4)
            col = int(mouse_position.x()) // (WINDOW_WIDTH // 4)
            print(row, col)
            area = row * 4 + col + 1
            self.text_label.appendPlainText(f"Area: {area}")


            if START_SEND_PACKET:
                MOUSE_CLICKED = True

    # Added for 3d coordinates, Jason, 11 April 2024
    def on_send_packed_finished(self):
        global SENDING_PACKET
        print('on_send_packed_finished')
        SENDING_PACKET = False

    # Added for 3d coordinates, Jason, 11 April 2024
    def send_3d_point(self):
        global SENDING_PACKET

        if not self.adminRole:
            return

        if not hasattr(self, 'mouse_press_timer'):
            self.mouse_press_timer = QTimer()
            self.mouse_press_timer.setSingleShot(True)

        if not SENDING_PACKET:
            SENDING_PACKET = True
            self.thread_send_packet = QThread()
            # Disabled send packet
            self.thread_send_packet.run = self.sendPacket
            self.thread_send_packet.finished.connect(self.on_send_packed_finished)
            print("self.targetPos: ", self.targetPos)
            print('Start sending 3D point packet')
            self.thread_send_packet.start()
            
        else:
            print('Wait for sending the last packet...')
            self.mouse_press_timer.start(2000)

    def saving_thread_finished(self):
        del self.video_saving_thread

    def combine_thread_finished(self):
        print('finished')
        if self.video_thread.d435:
            self.video_thread.d435.resume_thread()
        elif self.video_thread.cam:
            self.video_thread.cam.resume_thread()
        self.video_thread.resume_thread()
        self.show_combine_finished_dialog()
        del self.combine_thread

    def record_button_debounce_finished(self):
        self.record_button.setEnabled(True)

    def record_button_clicked(self):
        global START_RECORDING, VIDEO_NAME, AUDIO_NAME, OUTPUT_NAME
        self.RECORDING = not self.RECORDING
        if self.RECORDING:
            START_RECORDING = True
            self.record_button.setIcon(self.icon_stop_record)
            self.record_button.setEnabled(False)

            if not hasattr(self, 'record_button_debounce_timer'):
                self.record_button_debounce_timer = QTimer()
                self.record_button_debounce_timer.setSingleShot(True)
                self.record_button_debounce_timer.timeout.connect(self.record_button_debounce_finished)
            self.record_button_debounce_timer.start(2000)
            # self.record_button.setStyleSheet("background-color:red ; color :white ;border-width: 4px;border-radius: 20px;")

            self.text_label.appendPlainText('Status: Recording')
            start_recording(self)
            
            self.video_saving_thread = VideoSavingThread(self.video_thread.cam, self.video_thread.d435, App.CAM_DISPLAY_WIDTH, App.CAM_DISPLAY_HEIGHT)
            self.video_saving_thread.finished.connect(self.saving_thread_finished)
            self.video_saving_thread.start()
        else:
            START_RECORDING = False

            # self.record_button.setStyleSheet(BUTTON_STYLE_RED)
            self.record_button.setIcon(self.icon_start_record)
            self.text_label.appendPlainText('Status: Not Recording')
            self.record_button.setEnabled(False)
            self.record_button_debounce_timer.start(2000)

            self.audio_thread.requestInterruption()

            # revised[combine video, audio],Jason,17 May 2024
            self.combine_thread = VideoAudioThread(VIDEO_NAME,AUDIO_NAME,OUTPUT_NAME, self.logger)
            self.combine_thread.start_writing.connect(self.show_progress_dialog)
            self.combine_thread.start_writing.connect(self.video_thread.pause_thread)
            if self.video_thread.d435:
                self.combine_thread.start_writing.connect(self.video_thread.d435.pause_thread)
            elif self.video_thread.cam:
                self.combine_thread.start_writing.connect(self.video_thread.cam.pause_thread)
            self.combine_thread.finished.connect(self.combine_thread_finished)
            self.logger.signal_emitter.percentage_changed_signal.connect(self.update_progress_dialog_percentage)
            self.logger.signal_emitter.text_changed_signal.connect(self.update_progress_dialog_text)
            # disable combine, Jason, 20 May 2024
            self.combine_thread.start()

    def mic_on_off_button_clicked(self):
        self.MIC_ON = not self.MIC_ON
        if self.MIC_ON == True:
            self.mic_on_off_button.setIcon(self.icon_micOn)
            #enable slider_mic_vol,Brian,28 Feb 2024
            self.slider_mic_vol.setEnabled(True)
            self.audio_inCtrl.unmute()
            
        else:
            self.mic_on_off_button.setIcon(self.icon_micOff)
            #disalbe slider_mic_vol,Brian,28 Feb 2024
            self.slider_mic_vol.setEnabled(False)
            self.audio_inCtrl.mute()

    def sound_on_off_button_clicked(self):
        self.SOUND_ON = not self.SOUND_ON
        if self.SOUND_ON:
            self.sound_on_off_button.setIcon(self.icon_soundOn)
            #enable slider_sound_vol,Brian,28 Feb 2024
            self.slider_sound_vol.setEnabled(True)
            self.audio_outCtrl.unmute()
        else:
            self.sound_on_off_button.setIcon(self.icon_soundOff)
            #disalbe slider_sound_vol,Brian,28 Feb 2024
            self.slider_sound_vol.setEnabled(False)
            self.audio_outCtrl.mute()

    def showAlert(self,sTitle,sMsg,icon):
        message_box = QMessageBox()
        message_box.setWindowTitle(sTitle)
        message_box.setWindowIcon(icon)
        message_box.setIcon(QMessageBox.Information)
        message_box.setText(sMsg)
        message_box.addButton(QMessageBox.Ok)
        message_box.exec_()

    def show_combine_finished_dialog(self):
        self.video_combine_finished_dialog = QDialog(self)
        self.video_combine_finished_dialog.setWindowTitle('Video')
        layout = QVBoxLayout()
        message = QLabel('Finished generating video ' + OUTPUT_NAME)
        layout.addWidget(message)
        self.video_combine_finished_dialog.setLayout(layout)
        self.video_combine_finished_dialog.exec_()


    def show_progress_dialog(self):
        print('show progress dialog')
        self.video_progress_dialog = QProgressDialog(self)
        self.video_progress_dialog.setWindowTitle("Video")  
        self.video_progress_dialog.setLabelText("Generating video " + OUTPUT_NAME)
        self.video_progress_dialog.setCancelButton(None)
        self.video_progress_dialog.setWindowFlags(Qt.Window | Qt.WindowType.WindowTitleHint | Qt.WindowType.CustomizeWindowHint)
        self.video_progress_dialog.setMinimumDuration(5)
        self.video_progress_dialog.setWindowModality(Qt.WindowModal)
        self.video_progress_dialog.setRange(0,100)
        # self.video_progress_dialog.setGeometry(0, 0, 600, 100)

    @pyqtSlot(float)
    def update_progress_dialog_percentage(self, percentage):
        # print('u')
        if not self.video_progress_dialog.wasCanceled():
            self.video_progress_dialog.setValue(round(percentage))

    @pyqtSlot(str)
    def update_progress_dialog_text(self, text):
        if not self.video_progress_dialog.wasCanceled():
            self.video_progress_dialog.setLabelText(text + " Generating video " + OUTPUT_NAME)

    def exit_app(self):
        global dataLogger
        dataLogger.stop_logging()

        # stop all the timers
        if hasattr(self, 'mouse_press_timer'):
            self.mouse_press_timer.stop()
        exit()


if __name__ == "__main__":

    # Check if the folder exists
    if not os.path.exists('log'):
        # Create the folder
        os.makedirs('log')
        print("log folder created successfully.")
    else:
        print("log folder already exists.")

    # initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataLogger = DataLogger(log_interval=1, file_path="log/%s_sys.log" %(timestamp))  # Specify the data file path
    dataLogger.start_logging()
    dataLogger.add_data('logger started...')

    # Create necessary DIR
    check_folder_existence(CURRENT_PATH+VIDEO_SAVE_DIRECTORY)
    check_folder_existence(CURRENT_PATH+VIDEO_SAVE_DIRECTORY+VIDEO_DATE)
    check_folder_existence(CURRENT_PATH+AUDIO_PATH+"\\"+VIDEO_DATE)
    check_folder_existence(CURRENT_PATH+OUTPUT_PATH+"\\"+VIDEO_DATE)

    # os.environ["QT_ENABLE_HIGHDPI_SCALING"]   = "2"
    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0.5"
    # os.environ["QT_SCALE_FACTOR"]             = "2"
    app = QApplication(sys.argv)

    # app.setAttribute(Qt.AA_DisableHighDpiScaling)
    # print(f"Using AA_DisableHighDpiScaling > {QApplication.testAttribute(Qt.AA_DisableHighDpiScaling)}")
    # print(f"Using AA_UseHighDpiPixmaps    > {QApplication.testAttribute(Qt.AA_UseHighDpiPixmaps)}")
    try:
        a = App()
        a.showMaximized()
        # print("w, h: ", a.image_label.width(), a.image_label.height())
        # print("x, y: ", a.image_label.x(), a.image_label.y())
        app.aboutToQuit.connect(dataLogger.stop_logging)
        app.aboutToQuit.connect(a.cleanUp)
        exit(app.exec_())
    except Exception as e:
        # revised[add e and write to log if exception was caught],Brian,24 May 2024
        print(repr(e))
        dataLogger.stop_logging()

        
