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

from Widget_library import *
from Event import *
from Enum_library import  *
from Delay_Transmission import *

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
DEBUG = True

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
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(camera_names)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout = QVBoxLayout()
        layout.addWidget(self.camera_combo)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_selected_camera_name(self):
        return self.camera_combo.currentIndex()


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
    
    def __init__(self, arg1):
        super().__init__()
        self.camera_index = arg1

    def run(self):
        global START_RECORDING,VIDEO_NAME,OUTPUT_NAME,AUDIO_NAME

        
        
        # capture from web cam
        i = 0
        print(self.camera_index)
        cap = cv2.VideoCapture(self.camera_index)
        
        cap.set(3, WINDOW_WIDTH)
        cap.set(4, WINDOW_HEIGHT)
        video_width = int(cap.get(3))
        video_height = int(cap.get(4))

        print("Input ratio : ",video_height,video_width)
        while True:
            ret, self.cv_img = cap.read()
            if DEBUG == True:
                cv2.line(self.cv_img,(int(video_width/4),0) ,(int(video_width/4),int(video_height)) , (255, 100,  15), 2)
                cv2.line(self.cv_img,(int(video_width*2/4),0) ,(int(video_width*2/4),int(video_height)) , (255, 100,  15), 2)
                cv2.line(self.cv_img,(int(video_width*3/4),0) ,(int(video_width*3/4),int(video_height)) , (255, 100,  15), 2)

                cv2.line(self.cv_img,(0,int(video_height/4)) ,(int(video_width),int(video_height/4)) , (255, 100,  15), 2)
                cv2.line(self.cv_img,(0,int(video_height*2/4)) ,(int(video_width),int(video_height*2/4)) , (255, 100,  15), 2)
                cv2.line(self.cv_img,(0,int(video_height*3/4)) ,(int(video_width),int(video_height*3/4)) , (255, 100,  15), 2)
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
        self.setWindowTitle("ISD Mic array Project")
        self.setStyleSheet("background-color:lightgreen")
        self.index = APP_PAGE.MAIN.value
        self.LPF_Select = Frequency_Selection.LPF_FULL.value
        self.RECORDING = False

        # create the label that holds the image
        self.image_label = Create_ImageWindow()

        self.ExitButton = Create_Button("Exit",lambda:exit(),BUTTON_STYLE)
        self.SettingButton = Create_Button("Setting",lambda:switchPage(self,APP_PAGE.SETTING.value),BUTTON_STYLE)
        self.RecordButton = Create_Button("Record",self.Record_clicked,BUTTON_STYLE_RED)
        self.text_label = Create_Label()

        #Setting Up Main Page 
        self.MainPage = QGridLayout()
        self.MainPage.setContentsMargins(0,0,0,0)
        self.MainPage.setHorizontalSpacing(0)  # Set horizontal spacing to zero   
        self.MainPage.setVerticalSpacing(0)
        self.MainPage.addWidget(self.image_label,0,0,alignment=Qt.AlignCenter)
        self.MainPage.setHorizontalSpacing(0)  # Set horizontal spacing to zero   
        self.MainPage.setVerticalSpacing(0)  # Set horizontal spacing to zero   
        
        
        self.MainPage_button = QHBoxLayout()
        self.MainPage_button.addWidget(self.ExitButton,alignment=Qt.AlignLeft)
        self.MainPage_button.addWidget(self.RecordButton)
        self.MainPage_button.addWidget(self.SettingButton,alignment=Qt.AlignRight)
        self.MainPage_button_widget = QWidget()
        self.MainPage_button_widget.setLayout(self.MainPage_button)
        self.MainPage_button_widget.setFixedSize(WINDOW_WIDTH,BUTTON_BAR_HEIGHT)
        self.MainPage_button_widget.setStyleSheet("background-color:transparent")
        
        if DEBUG == True:
            self.MainPage.addWidget(self.text_label,0,0,alignment=Qt.AlignRight)
            self.MainPage.addWidget(self.MainPage_button_widget,1,0,1,2)
        else:
            self.MainPage.addWidget(self.MainPage_button_widget,0,0,alignment=Qt.AlignBottom)
            
        #Setting up Setting page for LPF and Gain and Voulme 
        self.GainLabel = QLabel("Mic Array Channel Gain  :")
        self.GainLabel.setFont(BUTTON_FONT)
        self.VolumeLabel = QLabel("Mic Array Digital Volume :")
        self.VolumeLabel.setFont(BUTTON_FONT)
        self.GainFader= Create_Slider(-12,12,0,1,SLIDER_STYLE,update_label)
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
        self.SettingPage.addWidget(self.GainLabel,1,0)
        self.SettingPage.addWidget(self.GainFader,1,1,1,5)
        self.SettingPage.addWidget(self.VolumeLabel,2,0)
        self.SettingPage.addWidget(self.VolumeFader,2,1,1,5)
        self.SettingPage.addWidget(self.FilterSelectLabel,3,0)
        self.SettingPage.addWidget(self.CheckBox_6kHz,3,1)
        self.SettingPage.addWidget(self.CheckBox_12kHz,3,2)
        self.SettingPage.addWidget(self.CheckBox_18kHz,3,3)
        self.SettingPage.addWidget(self.CheckBox_Full,3,4)

        self.SettingPage.addWidget(self.ApplyButton,4,0,1,4,alignment=Qt.AlignRight)
        self.SettingPage.addWidget(self.BackButton,4,1,1,4,alignment=Qt.AlignRight)
                
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
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.stacked_widget)
        # set the vbox layout as the widgets layout
        self.setLayout(self.layout)
        self.setFixedSize(WINDOW_WIDTH,WINDOW_HEIGHT)
      
        # self.setGeometry(QApplication.screens()[self.ScreenNumber].geometry())
        # self.showFullScreen()

        # pop up dialog to select video first
        dialog = CameraSelectionDialog(getCameraList())

        # Display the dialog and get the selected camera name
        if dialog.exec_() == QDialog.Accepted:
            self.selected_camera_index = dialog.get_selected_camera_name()
        else:
            print("Exit now...")
            exit()

        # create the video capture thread
        self.video_thread = VideoThread(self.selected_camera_index)
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
        p = convert_to_Qt_format.scaled(WINDOW_WIDTH, WINDOW_HEIGHT, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def mousePressEvent(self, event):
        # Handle mouse press events
        mouse_position = event.pos()
        self.text_label.appendPlainText(f"Clicked on [{mouse_position.x()},{mouse_position.y()}]")
        row = int(mouse_position.y()) // (WINDOW_HEIGHT // 4 )
        col =int( mouse_position.x()) // (WINDOW_WIDTH // 4)
        print(row , col )
        area = row*4 + col +1
        self.text_label.appendPlainText(f"Area: {area}") 
        # message = create_and_send_packet(HOST,PORT, area.to_bytes( 2, byteorder='big'))
        Test_delay_function()

    
    def Record_clicked(self):
        global START_RECORDING,VIDEO_NAME,AUDIO_NAME,OUTPUT_NAME
        self.RECORDING = not self.RECORDING
        if self.RECORDING == True:
            self.RecordButton.setStyleSheet("background-color:red ; color :white ;border-width: 4px;border-radius: 20px;")

            self.text_label.appendPlainText("Recording")
            start_recording(self)
            
        else:
            self.RecordButton.setStyleSheet(BUTTON_STYLE_RED)
            self.text_label.appendPlainText('Status: Not Recording')
            START_RECORDING=False
            self.audio_thread.requestInterruption()
            self.combine_thread = VideoAudioThread(VIDEO_NAME,AUDIO_NAME,OUTPUT_NAME)
            self.combine_thread.start()        
    
if __name__=="__main__":
    
# Create necessary DIR
    check_folder_existence(CURRENT_PATH+VIDEO_SAVE_DIRECTORY)
    check_folder_existence(CURRENT_PATH+VIDEO_SAVE_DIRECTORY+VIDEO_DATE)
    check_folder_existence(CURRENT_PATH+AUDIO_PATH+"\\"+VIDEO_DATE)
    check_folder_existence(CURRENT_PATH+OUTPUT_PATH+"\\"+VIDEO_DATE)
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())