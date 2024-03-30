from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, IMMDeviceEnumerator, EDataFlow, ERole
from pycaw.constants import CLSID_MMDeviceEnumerator
from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox, QComboBox, QDialog, QDialogButtonBox, QVBoxLayout, QPushButton
from PyQt5.QtMultimedia import QAudioDeviceInfo, QAudio, QAudioInput, QAudioFormat

import time
import sounddevice as sd
import soundfile as sf
import queue
import numpy as np
import wavio
# from threading import Thread
import threading
import multiprocessing
from ctypes import POINTER, cast
import comtypes
from comtypes import CLSCTX_ALL




class AudioDeviceDialog(QDialog):

    def __init__(self, input_devices, output_devices, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Audio Devices")

        layout = QVBoxLayout(self)

        # Create input device combo box
        self.input_combo = QComboBox()
        self.input_combo.addItems(input_devices)
        layout.addWidget(self.input_combo)

        # Create output device combo box
        self.output_combo = QComboBox()
        self.output_combo.addItems(output_devices)
        layout.addWidget(self.output_combo)

        # # Create start recording button
        # self.record_button = QPushButton("Start Recording")
        # self.record_button.clicked.connect(self.start_recording)
        # layout.addWidget(self.record_button)

        # Create dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Initialize recording variables
        self.recording = False
        self.frames = []

    def start_recording(self):
        if not self.recording:
            self.record_button.setText("Stop...")
            self.recording = True
            self.frames = []
            input_device = self.input_combo.currentText()

            def audio_callback(indata, frames, time, status):
                if self.recording:
                    self.frames.append(indata.copy())

            # Start recording audio in a separate thread
            def recording_thread():
                # Open the stream and start recording
                with sd.InputStream(device=input_device.split(',')[1],
                                    channels=2,
                                    callback=audio_callback):
                    # Wait for ten seconds or until recording is stopped
                    time.sleep(10)
                # Save the recorded audio as a WAV file
                if self.frames:
                    filename = "recorded_audio.wav"
                    with sf.SoundFile(filename,
                                      mode='w',
                                      samplerate=44100,
                                      channels=2) as file:
                        for frame in self.frames:
                            file.write(frame)
                        print(f"Recording saved as {filename}")
                else:
                    print("No audio recorded.")
                # Reset recording variables
                self.recording = False
                self.frames = []

            # Start the recording thread
            self.thread = threading.Thread(target=recording_thread)
            self.thread.start()
        else:
            # stop thread
            self.thread.stop()
            self.recording = False
            self.record_button.setText("Start Recording")

    def get_selected_devices(self):
        input_device = self.input_combo.currentText()
        output_device = self.output_combo.currentText()
        return input_device, output_device




class MyAudioUtilities(AudioUtilities):
    @staticmethod
    def GetSpeaker(id=None):
        device_enumerator = comtypes.CoCreateInstance(
            CLSID_MMDeviceEnumerator,
            IMMDeviceEnumerator,
            comtypes.CLSCTX_INPROC_SERVER)
        if id is not None:
            speakers = device_enumerator.GetDevice(id)
        else:
            speakers = device_enumerator.GetDefaultAudioEndpoint(EDataFlow.eRender.value, ERole.eMultimedia.value)
        return speakers

class AudioController:

    def __init__(self, process_name, input_device, output_device):
        self.process_name = process_name
        self.volume = self.process_volume() if self.process_volume() else 1.0
        self.prev_volume = 0.0
        self.input_device = input_device
        self.output_device = output_device
        # self.sample_rate = 44100    # Hz
        self.sample_rate = 48000
        self.recording = False
        # self.audio_buffer = []      # Buffer to store recorded audio data
        self.audio_record_queue = queue.Queue()
        self.filename = "recorded_audio.wav"
        self.thread = None

        devicelist = MyAudioUtilities.GetAllDevices()
        for device in devicelist:
            # simply select the first active speaker device?,Brian,30 Mar 2024
            if "Speaker" in str(device) and device.state.value==1:
                mixer_output = device
                break
        
        devices = MyAudioUtilities.GetSpeaker(mixer_output.id)
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.selectedSpeaker = cast(interface, POINTER(IAudioEndpointVolume))
        print("volume.GetMute(): %s" % self.selectedSpeaker.GetMute())
        print("volume.GetMasterVolumeLevel(): %s" % self.selectedSpeaker.GetMasterVolumeLevel())
        print("volume.GetVolumeRange(): (%s, %s, %s)" % self.selectedSpeaker.GetVolumeRange())
        # print("volume.SetMasterVolumeLevel()")
        # self.selectedSpeaker.SetMasterVolumeLevel(-5.0, None)
        # print("volume.GetMasterVolumeLevel(): %s" % self.selectedSpeaker.GetMasterVolumeLevel())

    
        

    def listAllSections(self):
        sessions = AudioUtilities.GetAllSessions()
        print('number of sessions=%d' % (len(sessions)))
        for session in sessions:
            interface = session.SimpleAudioVolume
            print("session: ", session, "\tinterface: ", interface)

    def mute(self):
        self.prev_volume = self.volume
        self.volume = 0.0
        self.set_volume(0)
        

    def unmute(self):
        self.volume = self.prev_volume
        self.set_volume(self.volume)

    def process_volume(self):
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            interface = session.SimpleAudioVolume
            if session.Process and session.Process.name() == self.process_name:
                print("Volume:", interface.GetMasterVolume(),self.process_name)  # debug
                return interface.GetMasterVolume()

    def set_volume(self, volumeVal):
        # Get default audio device using PyCAW
        # devices = AudioUtilities.GetSpeakers()
        # interface = devices.Activate(
        #     IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        # volume = cast(interface, POINTER(IAudioEndpointVolume))
        # # Get current volume 
        # currentVolumeDb = volume.GetMasterVolumeLevel()
        # print(currentVolumeDb)

        # input value is between 0 and 100
        volumeVal/=100.

        # convert volumeVal to dB
        volumeVal = min(1.0,max(0.0,volumeVal)) # make sure that volumeVal is in the correct range
        if volumeVal>0.:
            volumeDB = 20*np.log10(volumeVal)/0.6
            if volumeDB<-65.25:
                volumeDB = -65.25
        else:
            volumeDB = -65.25

        print(volumeVal,volumeDB)
        self.selectedSpeaker.SetMasterVolumeLevel(volumeDB, None)

        # self.volume = min(1.0, max(0.0, decibels/100.0))
        # print("New Volume: ", self.volume)
        # sessions = AudioUtilities.GetAllSessions()
        # for session in sessions:
        #     interface = session.SimpleAudioVolume
        #     if session.Process and session.Process.name() == self.process_name:
        #         print("Master Volume:", interface.GetMasterVolume())  # debug
                # only set volume in the range 0.0 to 1.0
                # self.volume = min(1.0, max(0.0, decibels/100.))             
                # interface.SetMasterVolume(self.volume, None)
                # print("Volume set to", self.volume)  # debug

    def decrease_volume(self, decibels):
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            interface = session.SimpleAudioVolume
            if session.Process and session.Process.name() == self.process_name:
                # 0.0 is the min value, reduce by decibels
                self.volume = max(0.0, self.volume - decibels)
                interface.SetMasterVolume(self.volume, None)
                print("Volume reduced to", self.volume)  # debug

    def increase_volume(self, decibels):
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            interface = session.SimpleAudioVolume
            if session.Process and session.Process.name() == self.process_name:
                # 1.0 is the max value, raise by decibels
                self.volume = min(1.0, self.volume + decibels)
                interface.SetMasterVolume(self.volume, None)
                print("Volume raised to", self.volume)  # debug

    # def callback(self, indata, frames, time, status):
    #     if status:
    #         print(status)
    #     print("callback")
    #     # self.audio_record_queue.put(indata.copy())
    #     # mic 1 - host, mic 2 - audience
    #     # audio_record_queue - save to .wav file, includes both host and audience
    #     # audio_output_queue - output to speaker for the audience
    #     self.audio_record_queue.put(indata.copy())
    #     # self.audio_output_queue.put(indata[:, [0]].copy())
    #     self.audio_output_queue.put(indata.copy())

    def callback(self, indata, outdata, frames, time, status):
        if status:
            print(status)
        if self.recording:
            self.audio_record_queue.put(indata.copy())
        outdata[:] = indata[:] * self.volume

    # def start_stop_recording(self, recording):
    #     self.recording = recording
    #     if self.recording:
    #         def recording_thread():
    #             with sf.SoundFile(self.filename, mode='w', samplerate=self.sample_rate, channels=2, subtype=None) as file:
    #                 # default input & output devices
    #                 with sd.Stream(samplerate=self.sample_rate, channels=2, callback=self.file_output_callback):
    #                     while self.recording:
    #                         audio_output = self.audio_record_queue.get()
    #                         file.write(audio_output)

    #         if not self.file_output_thread:
    #             self.file_output_thread = Thread(target=recording_thread)
    #             self.file_output_thread.start()

    def start_stop_recording(self, recording):
        print("start_stop_recording: ", recording)
        self.recording = recording
        if self.recording:
            def recording_thread():
                with sf.SoundFile(self.filename, mode='w', samplerate=self.sample_rate, channels=2, subtype=None) as file:
                    # default input & output devices
                    with sd.Stream(samplerate=self.sample_rate, channels=2, callback=self.callback):
                        # while True:
                            while self.recording:
                                audio_output = self.audio_record_queue.get()
                                file.write(audio_output)
                            # time.sleep(100)
                
            if not self.thread:
                print("start thread")
                self.thread = threading.Thread(target=recording_thread)
                self.thread.start()

    # def start_stop_recording(self, recording):
    #     print("start_stop_recording ", recording)
    #     self.recording = recording
    #     # if self.recording:
    #     def recording_thread():
    #         with sf.SoundFile(self.filename, mode='w', samplerate=self.sample_rate, channels=2, subtype=None) as file:
    #             # with sd.InputStream(samplerate=self.sample_rate, device=self.input_device.split(',')[1], channels=1, callback=self.callback):
    #             # default input & output devices
    #             with sd.Stream(samplerate=self.sample_rate, channels=2, callback=self.callback):
    #                 while True:
    #                     print("T self.recording: ", self.recording)
    #                     while self.recording:
    #                         audio_output = self.audio_record_queue.get()
    #                         file.write(audio_output)
    #                         # file.write(self.audio_record_queue.get())
    #                     time.sleep(100)
                
    #     if not self.thread:
    #         print("start thread")
    #         self.thread = Thread(target=recording_thread)
    #         self.thread.start()




