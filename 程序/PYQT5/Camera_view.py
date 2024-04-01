from pywinauto import Desktop  # add this to handle UI scaling issue
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
import wavio  #pip3 install wavio
from datetime import datetime

import time
from sys import exit

from Widget_library import *
from Event import *
from Enum_library import *
from Delay_Transmission import *
from audio_controller import AudioDeviceDialog, AudioController, MyAudioUtilities
from utility import audio_dev_to_str

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
        available_cameras.append(camera.description())
    return available_cameras


class CameraSelectionDialog(QDialog):

    def __init__(self, camera_names):
        super().__init__()

        self.setWindowTitle("Camera Selection")
        self.setFixedWidth(550)
        self.setFixedHeight(200)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(
                resource_path("mic_double_FILL0_wght400_GRAD0_opsz24.svg")),
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(camera_names)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok
                                      | QDialogButtonBox.Cancel)
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
        combine_video_audio(self.video_path, self.audio_path, self.output_path)


# Video Thread
class VideoThread(QThread):

    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, arg1):
        super().__init__()
        self.camera_index = arg1

    def run(self):
        global START_RECORDING, VIDEO_NAME, OUTPUT_NAME, AUDIO_NAME

        # capture from web cam
        i = 0
        print(self.camera_index)
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        print(cap.get(3), cap.get(4))

        video_width = int(cap.get(3))
        video_height = int(cap.get(4))
        print("Input ratio : ", video_height, video_width)
        while True:
            ret, self.cv_img = cap.read()
            if DEBUG:
                cv2.line(self.cv_img, (int(video_width / 4), 0),
                         (int(video_width / 4), int(video_height + 95)),
                         (255, 100, 15), 2)
                cv2.line(self.cv_img, (int(video_width * 2 / 4), 0),
                         (int(video_width * 2 / 4), int(video_height + 95)),
                         (255, 100, 15), 2)
                cv2.line(self.cv_img, (int(video_width * 3 / 4), 0),
                         (int(video_width * 3 / 4), int(video_height + 95)),
                         (255, 100, 15), 2)

                cv2.line(self.cv_img, (0, int(video_height / 4 + 95)),
                         (int(video_width), int(video_height / 4 + 95)),
                         (255, 100, 15), 2)
                cv2.line(self.cv_img, (0, int(video_height * 2 / 4 + 95)),
                         (int(video_width), int(video_height * 2 / 4 + 95)),
                         (255, 100, 15), 2)
                cv2.line(self.cv_img, (0, int(video_height * 3 / 4 + 95)),
                         (int(video_width), int(video_height * 3 / 4 + 95)),
                         (255, 100, 15), 2)
            if START_RECORDING:
                if (i == 0):
                    print("initiate Video")
                    current_datetime = datetime.now()
                    formatted_datetime = current_datetime.strftime(
                        "[%m-%d-%y]%H_%M_%S")
                    self.video_name = CURRENT_PATH + VIDEO_SAVE_DIRECTORY + VIDEO_DATE + "\\" + str(
                        formatted_datetime) + '.avi'
                    self.output_path = CURRENT_PATH + OUTPUT_PATH + VIDEO_DATE + "\\" + str(
                        formatted_datetime) + '.mp4'
                    print(self.video_name)
                    print(self.output_path)
                    VIDEO_NAME = self.video_name
                    OUTPUT_NAME = self.output_path
                    AUDIO_NAME = CURRENT_PATH + AUDIO_PATH + VIDEO_DATE + "\\" + str(
                        formatted_datetime) + '.wav'
                    FPS = 25
                    out = cv2.VideoWriter(
                        self.video_name,
                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS,
                        (video_width, video_height))
                    i += 1
                else:
                    out.write(self.cv_img)

            else:
                if (i > 0):
                    out.release()
                    i = 0
            if ret:
                self.change_pixmap_signal.emit(self.cv_img)


# Audio Thread
class AudioThread(QThread):

    def __init__(self, input_device):
        super().__init__()
        self.sample_rate = 48000    # Hz
        self.audio_buffer = []      # Buffer to store recorded audio data
        self.moving_window= []      # a moving window to get average audio level
        self.input_device = input_device # the input device index
        self.avgLevel = 0.
            

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
            self.audio_buffer.append(indata.copy())

        
        if START_RECORDING:
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("[%m-%d-%y]%H_%M_%S")

            with sd.InputStream(device=self.input_device,callback=callback,
                                channels=1,
                                samplerate=self.sample_rate):
                while not self.isInterruptionRequested():
                    self.msleep(100)  # Adjust the sleep interval based on your preference

            # Convert the buffer to a numpy array
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            audio_name = CURRENT_PATH + AUDIO_PATH + VIDEO_DATE + "\\" + str(
                formatted_datetime) + '.wav'
            AUDIO_NAME = audio_name
            # Save the audio data to a WAV file
            wavio.write(audio_name, audio_data, self.sample_rate, sampwidth=3)


class App(QWidget):

    def __init__(self):
        super().__init__()
        if (QDesktopWidget().screenCount() > 1):
            self.screen_number = 1
        else:
            self.screen_number = 0
        self.setWindowTitle("ISD UI Mockup — v0.1.2")
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

        # create the label that holds the image
        self.image_label = Create_ImageWindow()
        # self.image_label.setMinimumSize(QSize(640, 480))
        # self.image_label.setMaximumSize(QSize(1600, 900))
        # self.image_label.setFixedSize(QSize(FRAME_WIDTH, FRAME_HEIGHT))

        self.exit_button = Create_Button("Exit", lambda: exit(), BUTTON_STYLE)
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

        if DEBUG == True:
            self.main_page.addWidget(self.text_label,
                                     0,
                                     0,
                                     1,
                                     2,
                                     alignment=Qt.AlignRight)
            self.main_page.addWidget(self.main_page_button_widget, 1, 0, 1, 2)
        else:
            self.main_page.addWidget(self.main_page_button_widget, 1, 0, 1, 2,
                                     Qt.AlignCenter)

        #Setting up Setting page for LPF and Gain and Voulme
        self.gain_label = QLabel("Mic Array Channel Gain  :")
        # button_font = QFont("Arial",40)
        # button_font.setPixelSize(40)
        self.gain_label.setFont(BUTTON_FONT)

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
            BUTTON_STYLE)
        self.ApplyButton = Create_Button("Apply", lambda: exit(), BUTTON_STYLE)

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

        self.setting_page_widget = QWidget()
        self.setting_page_widget.setLayout(self.setting_page)
        self.setting_page_widget.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        

        self.main_page_widget = QWidget()
        self.main_page_widget.setLayout(self.main_page)
        self.main_page_widget.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.stacked_widget.setContentsMargins(0, 0, 0, 0)
        self.stacked_widget.addWidget(self.main_page_widget)
        self.stacked_widget.addWidget(self.setting_page_widget)

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
            self.selected_camera_index = camera_dialog.get_selected_camera_name()
        else:
            print("Exit now...")
            exit()

        # pop up audio_dialog to select audio device
            
        # revised[not using sd to get list of audio devices],Brian, 1 April 2024
        # devices = sd.query_devices()
        # input_devices = [
        #     '%d,%s' % (i, audio_dev_to_str(device))
        #     for i, device in enumerate(devices)
        #     if device['max_input_channels'] > 0
        # ]
        
        # output_devices = [
        #     '%d,%s' % (i, audio_dev_to_str(device))
        #     for i, device in enumerate(devices)
        #     if device['max_output_channels'] > 0
        # ]
        # audio_dialog = AudioDeviceDialog(input_devices, output_devices,self.windowIcon())

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
        else:
            print("Exit now...")
            exit()

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
        self.audio_outCtrl = AudioController(self.output_devid,'output','log')
        self.audio_outCtrl.listAllSections()
        print(self.audio_outCtrl.volume)

        # turn to zero volume at start
        self.audio_outCtrl.set_volume(0)
        self.slider_sound_vol.setValue(0)
        self.slider_sound_vol.valueChanged.connect(self.audio_outCtrl.set_volume)

        # create audio controller for the input device
        self.audio_inCtrl = AudioController(self.input_devid,'input','linear')

        # turn to zero volume at start
        self.audio_inCtrl.set_volume(0)
        self.slider_mic_vol.setValue(0)
        self.slider_mic_vol.valueChanged.connect(self.audio_inCtrl.set_volume)

        # create the video capture thread
        self.video_thread = VideoThread(self.selected_camera_index)
        self.audio_thread = AudioThread(self.input_device)
        
        # connect its signal to the update_image slot
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        # start video thread
        self.video_thread.start()
        # start audio thread
        self.audio_thread.start()

        
        
        
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

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
        # Handle mouse press events
        mouse_position = event.pos()
        self.text_label.appendPlainText(
            f"Clicked on [{mouse_position.x()},{mouse_position.y()}]")
        row = int(mouse_position.y()) // (WINDOW_HEIGHT // 4)
        col = int(mouse_position.x()) // (WINDOW_WIDTH // 4)
        print(row, col)
        area = row * 4 + col + 1
        self.text_label.appendPlainText(f"Area: {area}")
        # message = create_and_send_packet(HOST,PORT, area.to_bytes( 2, byteorder='big'))
        Test_delay_function()

    def record_button_clicked(self):
        global START_RECORDING, VIDEO_NAME, AUDIO_NAME, OUTPUT_NAME
        self.RECORDING = not self.RECORDING
        if self.RECORDING:
            
            self.record_button.setIcon(self.icon_stop_record)
            # self.record_button.setStyleSheet("background-color:red ; color :white ;border-width: 4px;border-radius: 20px;")

            self.text_label.appendPlainText('Status: Recording')
            #self.audio_outCtrl.start_stop_recording(self.RECORDING)
            start_recording(self)
            START_RECORDING = True

        else:
            # self.record_button.setStyleSheet(BUTTON_STYLE_RED)
            self.record_button.setIcon(self.icon_start_record)
            self.text_label.appendPlainText('Status: Not Recording')
            START_RECORDING = False
            #self.audio_outCtrl.start_stop_recording(self.RECORDING)
            self.audio_thread.requestInterruption()
            print("OUTPUT_NAME: ", OUTPUT_NAME)
            self.combine_thread = VideoAudioThread(VIDEO_NAME,AUDIO_NAME,OUTPUT_NAME)
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




if __name__ == "__main__":
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
    a = App()
    a.show()
    print("w, h: ", a.image_label.width(), a.image_label.height())
    print("x, y: ", a.image_label.x(), a.image_label.y())
    sys.exit(app.exec_())
