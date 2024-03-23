from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *

import sys
import cv2

from Style import *
import numpy as np
import sounddevice as sd 
import os
import wavio #pip3 install wavio
from datetime import datetime

import time
from sys import exit

from Widget_library import *
from Event import *
from Enum_library import  *
from Delay_Transmission import *


import RealSense_library as RS_lib
import pyrealsense2 as rs

# from pywinauto import Desktop
current_datetime = datetime.now()
# Format the date and time
formatted_datetime = current_datetime.strftime("%m-%d-%y")
# Define Directory Path
VIDEO_SAVE_DIRECTORY = "\Video"
AUDIO_PATH = "\Audio"
OUTPUT_PATH = "\Combined"
VIDEO_DATE = "\\" + formatted_datetime
AUDIO_NAME=""
VIDEO_NAME =""
OUTPUT_NAME=""

display_monitor= 0
CURRENT_PATH = os.getcwd()
START_RECORDING = False
MIC_ON = True
SOUND_ON = True
DEBUG = False
PIXEL_X = 0 
PIXEL_Y = 0
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        # base_path = os.path.abspath("./res")
        base_path = os.path.abspath("./程序/PYQT5/res")

    return os.path.join(base_path, relative_path)

def getCameraList():
    # Get available camera indices
    available_cameras=[]
    for i,camera in enumerate(QCameraInfo.availableCameras()):
        available_cameras.append(camera.description())
    return available_cameras


class CameraSelectionDialog(QDialog):
    def __init__(self, camera_names):
        super().__init__()

        self.setWindowTitle("Camera Selection")
        self.setFixedWidth(550)
        self.setFixedHeight(200)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(resource_path("mic_double_FILL0_wght400_GRAD0_opsz24.svg")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(camera_names)
        self.list = camera_names
        print(camera_names)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout = QVBoxLayout()
        layout.addWidget(self.camera_combo)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_selected_camera_name(self):
        # return self.camera_combo.currentIndex()
        return 2,"Depth Camera" #hardcode return depth camera on Eric PC 


# Combine As output thread 
class VideoAudioThread(QThread):
    finished = pyqtSignal()

    def __init__(self, video_path, audio_path, output_path):
        super().__init__()
        self.video_path = video_path
        self.audio_path = audio_path
        self.output_path = output_path

    def run(self):
        print("Combine Video")
        time.sleep(1)
        combine_video_audio(self.video_path, self.audio_path,self.output_path)


# Video Thread
class VideoThread(QThread):

    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, arg1,arg2):
        super().__init__()
        self.camera_index = arg1
        self.camera_name = arg2
    def run(self):
        global START_RECORDING,VIDEO_NAME,OUTPUT_NAME,AUDIO_NAME,depth_frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.
        font_color = (0, 0, 255)  # White color
        if "Depth" in self.camera_name:
            print("Using Depth camera")
            # capture from web cam
            i = 0
            # print(self.camera_index)

            pipeline = rs.pipeline()
            config = rs.config()
            # pc = rs.pointcloud()

            config.enable_stream(rs.stream.depth, RS_lib.DEPTH_CAM_WIDTH, RS_lib.DEPTH_CAM_HEIGHT, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, RS_lib.COLOR_CAM_WIDTH, RS_lib.COLOR_CAM_HEIGHT, rs.format.bgr8, 30)

            profile = pipeline.start(config)
            align = rs.align(rs.stream.color)

            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            color_intrinsics = color_profile.get_intrinsics()

            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            depth_intrinsics = depth_profile.get_intrinsics()
            print("depth_intrinsics: ", depth_intrinsics)
            w, h = depth_intrinsics.width, depth_intrinsics.height

            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print("Depth Scale is: " , depth_scale)
            # for visualization
            depth_min = 0.1 #meter
            depth_max = 15.0 #meter

            colorizer = rs.colorizer()
            colorizer.set_option(rs.option.visual_preset, 1) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
            colorizer.set_option(rs.option.min_distance, depth_min)
            colorizer.set_option(rs.option.max_distance, depth_max)



            cap = cv2.VideoCapture(self.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, RS_lib.COLOR_CAM_HEIGHT)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RS_lib.COLOR_CAM_HEIGHT)

            print(cap.get(3), cap.get(4))

            video_width = int(cap.get(3))
            video_height = int(cap.get(4))
            print("Input ratio : ",video_height,video_width)
            while True:
                
                # Get a frameset from the pipeline
                frameset = pipeline.wait_for_frames()

                # Align the depth frame to the color frame
                aligned_frameset = align.process(frameset)

                # Get the aligned depth and color frames
                depth_frame = aligned_frameset.get_depth_frame()
                color_frame = aligned_frameset.get_color_frame()

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                cv2.circle(color_image,(PIXEL_X,PIXEL_Y),1,font_color,cv2.LINE_AA)
                cv2.putText(color_image, str(depth_frame.get_distance(PIXEL_X,PIXEL_Y)), (0, 100), font, font_scale, font_color, 2, cv2.LINE_AA)
                self.cv_img = color_image

                # ret, self.cv_img = cap.read()
                if DEBUG == True:
                    cv2.line(self.cv_img,(int(video_width/4),0) ,(int(video_width/4),int(video_height+95)) , (255, 100,  15), 2)
                    cv2.line(self.cv_img,(int(video_width*2/4),0) ,(int(video_width*2/4),int(video_height+95)) , (255, 100,  15), 2)
                    cv2.line(self.cv_img,(int(video_width*3/4),0) ,(int(video_width*3/4),int(video_height+95)) , (255, 100,  15), 2)
                    
                    cv2.line(self.cv_img,(0,int(video_height/4 + 95)) ,(int(video_width),int(video_height/4 + 95)) , (255, 100,  15), 2)
                    cv2.line(self.cv_img,(0,int(video_height*2/4 + 95)) ,(int(video_width),int(video_height*2/4 + 95)) , (255, 100,  15), 2)
                    cv2.line(self.cv_img,(0,int(video_height*3/4 + 95)) ,(int(video_width),int(video_height*3/4 + 95)) , (255, 100,  15), 2)
                if (START_RECORDING == True):
                    if (i == 0):
                        print("initiate Video")
                        current_datetime = datetime.now()
                        formatted_datetime = current_datetime.strftime("[%m-%d-%y]%H_%M_%S")
                        self.video_name = CURRENT_PATH+VIDEO_SAVE_DIRECTORY+VIDEO_DATE + "\\"+str(formatted_datetime)+'.avi'
                        self.output_path = CURRENT_PATH+OUTPUT_PATH+VIDEO_DATE+ "\\"+str(formatted_datetime)+'.mp4'
                        VIDEO_NAME = self.video_name
                        OUTPUT_NAME=self.output_path
                        AUDIO_NAME = CURRENT_PATH+AUDIO_PATH+VIDEO_DATE+ "\\"+str(formatted_datetime)+'.wav'
                        FPS = 25
                        out = cv2.VideoWriter(self.video_name, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (video_width,video_height))
                        i+=1
                    else:
                        out.write(self.cv_img)
                    
                else:
                    if (i > 0 ):
                        out.release()
                        i=0
                # if ret:
                self.change_pixmap_signal.emit(self.cv_img)
            
        else:
        
            # capture from web cam
            i = 0
            print(self.camera_index)
            cap = cv2.VideoCapture(self.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            print(cap.get(3), cap.get(4))

            video_width = int(cap.get(3))
            video_height = int(cap.get(4))
            print("Input ratio : ",video_height,video_width)
            while True:
                ret, self.cv_img = cap.read()
                if DEBUG == True:
                    cv2.line(self.cv_img,(int(video_width/4),0) ,(int(video_width/4),int(video_height+95)) , (255, 100,  15), 2)
                    cv2.line(self.cv_img,(int(video_width*2/4),0) ,(int(video_width*2/4),int(video_height+95)) , (255, 100,  15), 2)
                    cv2.line(self.cv_img,(int(video_width*3/4),0) ,(int(video_width*3/4),int(video_height+95)) , (255, 100,  15), 2)
                    
                    cv2.line(self.cv_img,(0,int(video_height/4 + 95)) ,(int(video_width),int(video_height/4 + 95)) , (255, 100,  15), 2)
                    cv2.line(self.cv_img,(0,int(video_height*2/4 + 95)) ,(int(video_width),int(video_height*2/4 + 95)) , (255, 100,  15), 2)
                    cv2.line(self.cv_img,(0,int(video_height*3/4 + 95)) ,(int(video_width),int(video_height*3/4 + 95)) , (255, 100,  15), 2)
                if (START_RECORDING == True):
                    if (i == 0):
                        print("initiate Video")
                        current_datetime = datetime.now()
                        formatted_datetime = current_datetime.strftime("[%m-%d-%y]%H_%M_%S")
                        self.video_name = CURRENT_PATH+VIDEO_SAVE_DIRECTORY+VIDEO_DATE + "\\"+str(formatted_datetime)+'.avi'
                        self.output_path = CURRENT_PATH+OUTPUT_PATH+VIDEO_DATE+ "\\"+str(formatted_datetime)+'.mp4'
                        VIDEO_NAME = self.video_name
                        OUTPUT_NAME=self.output_path
                        AUDIO_NAME = CURRENT_PATH+AUDIO_PATH+VIDEO_DATE+ "\\"+str(formatted_datetime)+'.wav'
                        FPS = 25
                        out = cv2.VideoWriter(self.video_name, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (video_width,video_height))
                        i+=1
                    else:
                        out.write(self.cv_img)
                    
                else:
                    if (i > 0 ):
                        out.release()
                        i=0
                if ret:
                    self.change_pixmap_signal.emit(self.cv_img)
# Audio Thread
class AudioThread(QThread):
    def run(self):
        global START_RECORDING,AUDIO_NAME
        sample_rate = 48000  # Hz
        self.audio_buffer = []  # Buffer to store recorded audio data

        # Use a callback function for non-blocking audio recording
        def callback(indata, frames, time, status):
            if status:
                print(status)
            # print("Recording audio...")
            self.audio_buffer.append(indata.copy())


        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("[%m-%d-%y]%H_%M_%S")
        START_RECORDING = True
        with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
             while not self.isInterruptionRequested():
                self.msleep(100)  # Adjust the sleep interval based on your preference

        # Convert the buffer to a numpy array
        audio_data = np.concatenate(self.audio_buffer, axis=0)
        audio_name = CURRENT_PATH+AUDIO_PATH+VIDEO_DATE + "\\"+str(formatted_datetime)+'.wav'
        AUDIO_NAME=audio_name
        # Save the audio data to a WAV file
        wavio.write(audio_name, audio_data, sample_rate, sampwidth=3)



class App(QWidget):
    def __init__(self):
        super().__init__()
        if (QDesktopWidget().screenCount() >1):
            self.ScreenNumber = 1
        else:
            self.ScreenNumber = 0
        self.ScreenNumber = 0
        self.setWindowTitle("ISD UI Mockup — v0.1.2")
        # self.setStyleSheet("background-color:gray")
        self.setStyleSheet("background-color:lightgreen") 
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(resource_path("mic_double_FILL0_wght400_GRAD0_opsz24.svg")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        self.index = APP_PAGE.MAIN.value
        self.LPF_Select = Frequency_Selection.LPF_FULL.value
        self.RECORDING = False
        self.MIC_ON = MIC_ON
        self.SOUND_ON = SOUND_ON

        # create the label that holds the image
        self.image_label = Create_ImageWindow()
        self.image_label.setMinimumSize(QSize(640, 480))
        self.image_label.setMaximumSize(QSize(1600, 900))
        # self.image_label.setFixedSize(QSize(FRAME_WIDTH, FRAME_HEIGHT))

        self.ExitButton = Create_Button("Exit",lambda:exit(),BUTTON_STYLE)
        # self.SettingButton = Create_Button("Setting",lambda:switchPage(self,APP_PAGE.SETTING.value),BUTTON_STYLE)
        # self.RecordButton = Create_Button("Record",self.Record_clicked,BUTTON_STYLE_RED)
        
        # Setting Button
        self.SettingButton = Create_Button("",lambda:switchPage(self,APP_PAGE.SETTING.value), "QPushButton {\n"
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
        self.SettingButton.setText("")
        icon2 = QIcon(resource_path("settings_FILL0_wght400_GRAD0_opsz24.svg"))
        self.SettingButton.setIcon(icon2)
        self.SettingButton.setIconSize(QSize(65, 65))
        self.SettingButton.setObjectName("btnSettings")
        self.SettingButton.setFixedSize(NUM_ROUND_BUTTON_SIZE, NUM_ROUND_BUTTON_SIZE)

        # Record Button
        self.icon_startRecord = QIcon(resource_path("radio_button_checked_FILL0_wght400_GRAD0_opsz24.png"))
        self.icon_stopRecord  = QIcon(resource_path("stop_circle_FILL0_wght400_GRAD0_opsz24.png"))
        self.RecordButton = Create_Button("",self.Record_clicked,"QPushButton {\n"
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
        self.RecordButton.setIcon(self.icon_startRecord)
        self.RecordButton.setIconSize(QSize(70, 70))
        self.RecordButton.setObjectName("btnRecord")
        self.RecordButton.setFixedSize(NUM_ROUND_BUTTON_SIZE, NUM_ROUND_BUTTON_SIZE)

        # Mic On Off Button
        self.MicOnOffButton = Create_Button("", self.MicOnOff_clicked, "QPushButton {\n"
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
        self.MicOnOffButton.setText("")
        self.icon_micOn = QIcon(resource_path("mic_FILL0_wght400_GRAD0_opsz24.svg"))
        self.icon_micOff = QIcon(resource_path("mic_off_FILL0_wght400_GRAD0_opsz24.svg"))
        self.MicOnOffButton.setIconSize(QSize(70, 70))
        self.MicOnOffButton.setObjectName("btnMicOnOff")
        self.MicOnOffButton.setFixedSize(NUM_ROUND_BUTTON_SIZE, NUM_ROUND_BUTTON_SIZE)
        self.MicOnOffButton.setIcon(self.icon_micOn)

        # Volume On Off Button
        self.SoundOnOffButton = Create_Button("", self.SoundOnOff_clicked, "QPushButton {\n"
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
        self.SoundOnOffButton.setText("")
        self.icon_soundOn = QIcon(resource_path("volume_up_FILL0_wght400_GRAD0_opsz24.svg"))
        self.icon_soundOff = QIcon(resource_path("volume_off_FILL0_wght400_GRAD0_opsz24.svg"))
        self.SoundOnOffButton.setIcon(self.icon_soundOn)
        self.SoundOnOffButton.setIconSize(QSize(70, 70))
        self.SoundOnOffButton.setObjectName("btnSoundOnOff")
        self.SoundOnOffButton.setFixedSize(NUM_ROUND_BUTTON_SIZE, NUM_ROUND_BUTTON_SIZE)

        self.sliderMicVol = QSlider()
        self.sliderMicVol.setOrientation(Qt.Horizontal)
        self.sliderMicVol.setObjectName("sliderMicVol")
        self.sliderMicVol.setFixedSize(200, 40)
        self.sliderMicVol.setStyleSheet(SLIDER_STYLE_2)
        

        self.sliderSoundVol = QSlider()
        self.sliderSoundVol.setOrientation(Qt.Horizontal)
        self.sliderSoundVol.setObjectName("sliderSoundVol")
        self.sliderSoundVol.setFixedSize(200, 40)
        self.sliderSoundVol.setStyleSheet(SLIDER_STYLE_2)

        self.text_label = Create_Label()
        
        #Setting Up Main Page 
        self.MainPage = QGridLayout()
        self.MainPage.setContentsMargins(0,0,0,0)
        self.MainPage.setHorizontalSpacing(0)  # Set horizontal spacing to zero   
        self.MainPage.addWidget(self.image_label,0,0,1,2,alignment=Qt.AlignCenter)
        self.MainPage.setHorizontalSpacing(0)  # Set horizontal spacing to zero   
        self.MainPage.setVerticalSpacing(0)  # Set horizontal spacing to zero   
        

        self.MainPage_button = QGridLayout()
        self.MainPage_button.setContentsMargins(100, 0, 100, 50)
        self.MainPage_button.addWidget(self.ExitButton, 0, 0, Qt.AlignLeft)
        self.MainPage_button.addWidget(self.RecordButton, 0, 1, 1, 2, Qt.AlignCenter)
        self.button_slider_layout = QGridLayout()
        self.button_slider_layout.addWidget(self.sliderMicVol, 0, 0, 1, 1, Qt.AlignCenter)
        self.button_slider_layout.addWidget(self.MicOnOffButton, 1, 0, 1, 1, Qt.AlignCenter)
        self.button_slider_layout.addWidget(self.sliderSoundVol, 0, 1, 1, 1, Qt.AlignCenter)
        self.button_slider_layout.addWidget(self.SoundOnOffButton, 1, 1, 1, 1, Qt.AlignCenter)
        self.button_slider_layout.addWidget(self.SettingButton, 0, 3, 2, 1, Qt.AlignCenter)
        self.button_slider_layout.setSpacing(20)
        self.MainPage_button.addLayout(self.button_slider_layout, 0, 2, 1, 2, Qt.AlignRight)


        self.MainPage_button_widget = QWidget()
        self.MainPage_button_widget.setLayout(self.MainPage_button)
        self.MainPage_button_widget.setFixedSize(WINDOW_WIDTH,BUTTON_BAR_HEIGHT)
        # self.MainPage_button_widget.setStyleSheet("background-color:transparent; border:1px solid rgb(0, 255, 0);")
        
        if DEBUG == True:
            self.MainPage.addWidget(self.text_label,0,0,1,2,alignment=Qt.AlignRight)
            self.MainPage.addWidget(self.MainPage_button_widget,1,0,1,2)
        else:
            self.MainPage.addWidget(self.MainPage_button_widget,1,0,1,2, Qt.AlignCenter)
            
        #Setting up Setting page for LPF and Gain and Voulme 
        self.GainLabel = QLabel("Mic Array Channel Gain  :")
        # button_font = QFont("Arial",40)
        # button_font.setPixelSize(40)
        self.GainLabel.setFont(BUTTON_FONT)
    
        self.VolumeLabel = QLabel("Mic Array Digital Volume :")
        self.VolumeLabel.setFont(BUTTON_FONT)
        self.GainFader= Create_Slider(-12,12,0,1,SLIDER_STYLE_2,update_label)
        self.VolumeFader = Create_Slider(0,24,0,1,SLIDER_STYLE_2,update_label)
        self.FilterSelectLabel = QLabel("Mic Array Filter Select     :")
        self.FilterSelectLabel.setFont(BUTTON_FONT)

        self.CheckBox_6kHz = Create_RadioBotton('6khz',lambda:ToggleSelection(self,Frequency_Selection.LPF_6K.value))
        self.CheckBox_12kHz = Create_RadioBotton('12khz',lambda:ToggleSelection(self,Frequency_Selection.LPF_12K.value))
        self.CheckBox_18kHz = Create_RadioBotton('18khz',lambda:ToggleSelection(self,Frequency_Selection.LPF_18K.value))
        self.CheckBox_Full = Create_RadioBotton('Full Range',lambda:ToggleSelection(self,Frequency_Selection.LPF_FULL.value))
        self.BackButton = Create_Button("Back",lambda:switchPage(self,APP_PAGE.MAIN.value),BUTTON_STYLE)
        self.ApplyButton =Create_Button("Apply",lambda:exit(),BUTTON_STYLE)

        self.SettingPage = QGridLayout()
        self.SettingPage.addWidget(self.GainLabel,1,0,1,1)
        self.SettingPage.addWidget(self.GainFader,1,1,1,4, Qt.AlignHCenter)

        self.SettingPage.addWidget(self.VolumeLabel,2,0,1,1)
        self.SettingPage.addWidget(self.VolumeFader,2,1,1,4, Qt.AlignHCenter)

        self.SettingPage.addWidget(self.FilterSelectLabel,3,0,1,1)
        self.SettingPage.addWidget(self.CheckBox_6kHz,3,1,1,1,Qt.AlignHCenter)
        self.SettingPage.addWidget(self.CheckBox_12kHz,3,2,1,1,Qt.AlignHCenter)
        self.SettingPage.addWidget(self.CheckBox_18kHz,3,3,1,1,Qt.AlignHCenter)
        self.SettingPage.addWidget(self.CheckBox_Full,3,4,1,1,Qt.AlignHCenter)

        self.SettingPage.addWidget(self.ApplyButton,4,3,1,1, Qt.AlignHCenter)
        self.SettingPage.addWidget(self.BackButton,4,4,1,1, Qt.AlignHCenter)
                
        self.SettingPageWidget = QWidget()
        self.SettingPageWidget.setLayout(self.SettingPage)
        self.SettingPageWidget.setFixedSize(WINDOW_WIDTH,WINDOW_HEIGHT)

        self.MainPageWidget = QWidget()
        self.MainPageWidget.setLayout(self.MainPage)
        self.MainPageWidget.setFixedSize(WINDOW_WIDTH,WINDOW_HEIGHT)

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setFixedSize(WINDOW_WIDTH,WINDOW_HEIGHT)
        self.stacked_widget.setContentsMargins(0,0,0,0)
        self.stacked_widget.addWidget(self.MainPageWidget)
        self.stacked_widget.addWidget(self.SettingPageWidget)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.stacked_widget)
        self.layout.setContentsMargins(0,50,0,0)
        self.layout.addWidget(self.stacked_widget)
        # set the vbox layout as the widgets layout
        self.setLayout(self.layout)
        self.setFixedSize(WINDOW_WIDTH,WINDOW_HEIGHT)
      
        self.setGeometry(QApplication.screens()[self.ScreenNumber].geometry())
        # self.showFullScreen()

        

        # pop up dialog to select video first
        dialog = CameraSelectionDialog(getCameraList())

        # Display the dialog and get the selected camera name
        if dialog.exec_() == QDialog.Accepted:
            self.selected_camera_index,self.Camera_name = dialog.get_selected_camera_name()
        else:
            print("Exit now...")
            exit()

        # create the video capture thread
        self.video_thread = VideoThread(self.selected_camera_index,self.Camera_name)
        self.audio_thread = AudioThread()
        # connect its signal to the update_image slot
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        
        # start the thread
        self.video_thread.start()
        
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        # p = convert_to_Qt_format.scaled(WINDOW_WIDTH, WINDOW_HEIGHT, Qt.KeepAspectRatio)
        p = convert_to_Qt_format.scaled(FRAME_WIDTH, FRAME_HEIGHT, Qt.KeepAspectRatio)
        # p = convert_to_Qt_format.scaled(FRAME_WIDTH, FRAME_HEIGHT, Qt.IgnoreAspectRatio)
        return QPixmap.fromImage(p)

    def mousePressEvent(self, event):
        global PIXEL_X,PIXEL_Y
        # Handle mouse press events
        mouse_position = event.pos()
        self.text_label.appendPlainText(f"Clicked on [{mouse_position.x()},{mouse_position.y()}]")
        PIXEL_X = mouse_position.x()
        PIXEL_Y =mouse_position.y()
        print(mouse_position.y(),mouse_position.x())

        print(depth_frame.get_distance(mouse_position.x(),mouse_position.y()))
        row = int(mouse_position.y()) // (WINDOW_HEIGHT // 4 )
        col =int( mouse_position.x()) // (WINDOW_WIDTH // 4)
        # print(row , col )
        area = row*4 + col +1
        self.text_label.appendPlainText(f"Area: {area}") 
        # message = create_and_send_packet(HOST,PORT, area.to_bytes( 2, byteorder='big'))
        Test_delay_function()

    
    def Record_clicked(self):
        global START_RECORDING,VIDEO_NAME,AUDIO_NAME,OUTPUT_NAME
        self.RECORDING = not self.RECORDING
        if self.RECORDING == True:
            self.RecordButton.setIcon(self.icon_stopRecord)
            # self.RecordButton.setStyleSheet("background-color:red ; color :white ;border-width: 4px;border-radius: 20px;")

            self.text_label.appendPlainText("Recording")
            #start_recording(self)
            
        else:
            # self.RecordButton.setStyleSheet(BUTTON_STYLE_RED)
            self.RecordButton.setIcon(self.icon_startRecord)
            self.text_label.appendPlainText('Status: Not Recording')
            START_RECORDING=False
            # self.audio_thread.requestInterruption()
            # print("OUTPUT_NAME: ", OUTPUT_NAME)
            # self.combine_thread = VideoAudioThread(VIDEO_NAME,AUDIO_NAME,OUTPUT_NAME)
            # self.combine_thread.start()

    def MicOnOff_clicked(self):
        self.MIC_ON = not self.MIC_ON
        if self.MIC_ON == True:
            self.MicOnOffButton.setIcon(self.icon_micOn)
            #enable sliderMicVol,Brian,28 Feb 2024
            self.sliderMicVol.setEnabled(True)
            
        else:
            self.MicOnOffButton.setIcon(self.icon_micOff)
            #disalbe sliderMicVol,Brian,28 Feb 2024
            self.sliderMicVol.setEnabled(False)
            
            
    
    def SoundOnOff_clicked(self):
        self.SOUND_ON = not self.SOUND_ON
        if self.SOUND_ON == True:
            self.SoundOnOffButton.setIcon(self.icon_soundOn)

            #enable sliderSoundVol,Brian,28 Feb 2024
            self.sliderSoundVol.setEnabled(True)
        else:
            self.SoundOnOffButton.setIcon(self.icon_soundOff)
            #disalbe sliderSoundVol,Brian,28 Feb 2024
            self.sliderSoundVol.setEnabled(False)
            

    
# if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
# if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

if __name__=="__main__":
# Create necessary DIR
    # removed,Brian,28 Feb 2024
    # check_folder_existence(CURRENT_PATH+VIDEO_SAVE_DIRECTORY)
    # check_folder_existence(CURRENT_PATH+VIDEO_SAVE_DIRECTORY+VIDEO_DATE)
    # check_folder_existence(CURRENT_PATH+AUDIO_PATH+"\\"+VIDEO_DATE)
    # check_folder_existence(CURRENT_PATH+OUTPUT_PATH+"\\"+VIDEO_DATE)
    
    # os.environ["QT_ENABLE_HIGHDPI_SCALING"]   = "2"
    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0.5"
    # os.environ["QT_SCALE_FACTOR"]             = "2"
    app = QApplication(sys.argv)
    
    # app.setAttribute(Qt.AA_DisableHighDpiScaling)
    # print(f"Using AA_DisableHighDpiScaling > {QApplication.testAttribute(Qt.AA_DisableHighDpiScaling)}")
    # print(f"Using AA_UseHighDpiPixmaps    > {QApplication.testAttribute(Qt.AA_UseHighDpiPixmaps)}")
    a = App()
    a.show()
    print("w, h: ", a.image_label.width(), a.image_label.height())
    print("x, y: ", a.image_label.x(), a.image_label.y())
    sys.exit(app.exec_())



