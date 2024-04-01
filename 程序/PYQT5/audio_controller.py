from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, IMMDeviceEnumerator, EDataFlow, ERole, DEVICE_STATE
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

    def __init__(self, input_devices, output_devices,icon, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Audio Devices")
        self.setWindowIcon(icon)
        self.setFixedWidth(800)

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
    def dev_to_str(device):
        return device.FriendlyName
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
    
    @staticmethod
    def GetDevice(id=None, default=0):
        device_enumerator = comtypes.CoCreateInstance(
            CLSID_MMDeviceEnumerator,
            IMMDeviceEnumerator,
            comtypes.CLSCTX_INPROC_SERVER)
        if id is not None:
            thisDevice = device_enumerator.GetDevice(id)
        else:
            if default == 0:
                # output
                thisDevice = device_enumerator.GetDefaultAudioEndpoint(EDataFlow.eRender.value, ERole.eMultimedia.value)
            else:
                # input
                thisDevice = device_enumerator.GetDefaultAudioEndpoint(EDataFlow.eCapture.value, ERole.eMultimedia.value)
        return thisDevice
    
    @staticmethod
    def GetAudioDevices(direction="in", State = DEVICE_STATE.ACTIVE.value):
        devices = []
        # for all use EDataFlow.eAll.value
        if direction == "in":
            Flow = EDataFlow.eCapture.value     # 1
        else:
            Flow = EDataFlow.eRender.value      # 0
        
        deviceEnumerator = comtypes.CoCreateInstance(
            CLSID_MMDeviceEnumerator,
            IMMDeviceEnumerator,
            comtypes.CLSCTX_INPROC_SERVER)
        if deviceEnumerator is None:
            return devices
        

        collection = deviceEnumerator.EnumAudioEndpoints(Flow, State)
        if collection is None:
            return devices

        count = collection.GetCount()
        for i in range(count):
            dev = collection.Item(i)
            if dev is not None:
                if not ": None" in str(AudioUtilities.CreateDevice(dev)):
                    devices.append(AudioUtilities.CreateDevice(dev))
        return devices

class AudioController:

    def __init__(self, deviceId, sType, sMapping):
        '''
        
        deviceId -- device uuid
        sType    -- 'input' or 'output' 
        sMapping -- audio level mapping 'linear' or 'log'

        '''
        self.volume = 0.0
        self.prev_volume = 0.0
        self.deviceId = deviceId
        self.devType  = sType
        self.mapping  = sMapping 
        self.sample_rate = 48000
        self.recording = False
        # self.audio_buffer = []      # Buffer to store recorded audio data
        self.audio_record_queue = queue.Queue()
        self.filename = "recorded_audio.wav"
        self.thread = None
        
        mixer_output = None
        # devices = MyAudioUtilities.GetAudioDevices(direction='in')
        # for dev in devices:
        #     print('ii--',dev,dev.id)
        # devices = MyAudioUtilities.GetAudioDevices(direction='out')
        # for dev in devices:
        #     print('oo--',dev,dev.id)
        # devicelist = MyAudioUtilities.GetAllDevices()
        # for device in devicelist:
        #     if device.state.value==1:
        #         print(device)
        #     if sType == 'output':
        #         # simply select the first active speaker device?,Brian,30 Mar 2024
        #         if "Speaker" in str(device) and device.state.value==1:
        #             mixer_output = device
        #             break
        #     elif sType == 'input':
        #         # simply select the first active speaker device?,Brian,30 Mar 2024
        #         if "Speaker" in str(device) and device.state.value==1:
        #             mixer_output = device
        #             break
        #     else:
        #         pass
        
        
        #myDevice = MyAudioUtilities.GetSpeaker(mixer_output.id)
        if sType=='input':
            myDevice = MyAudioUtilities.GetDevice(self.deviceId,default=1)
        else:
            myDevice = MyAudioUtilities.GetDevice(self.deviceId,default=0)

        interface = myDevice.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.selectedDevice = cast(interface, POINTER(IAudioEndpointVolume))
        print("volume.GetMute(): %s" % self.selectedDevice.GetMute())
        print("volume.GetMasterVolumeLevel(): %s" % self.selectedDevice.GetMasterVolumeLevel())
        print("volume.GetVolumeRange(): (%s, %s, %s)" % self.selectedDevice.GetVolumeRange())

        # get device dB range
        self.minDB,self.maxDB,_ = self.selectedDevice.GetVolumeRange()
        # print("volume.SetMasterVolumeLevel()")
        # self.selectedDevice.SetMasterVolumeLevel(-5.0, None)
        # print("volume.GetMasterVolumeLevel(): %s" % self.selectedDevice.GetMasterVolumeLevel())

    
        

    def listAllSections(self):
        sessions = AudioUtilities.GetAllSessions()
        print('number of sessions=%d' % (len(sessions)))
        for session in sessions:
            interface = session.SimpleAudioVolume
            print("session: ", session, "\tinterface: ", interface)

    def mute(self):
        self.prev_volume = self.volume
        self.set_volume(0)
        

    def unmute(self):
        # recover prev_volume
        self.volume = self.prev_volume
        self.set_volume(self.volume*100)

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

        # print(self.selectedDevice.GetMasterVolumeLevel())

        # input value is between 0 and 100
        volumeVal/=100.

        # save volumeVal to self.volume
        self.volume = volumeVal

        # convert volumeVal to dB
        volumeVal = min(1.0,max(0.0,volumeVal)) # make sure that volumeVal is in the correct range
        if volumeVal>0.:
            if self.mapping == 'linear':
                # simply map volumeVal linearly somewhere between self.minDB and self.maxDB
                volumeDB = self.minDB + volumeVal*(self.maxDB-self.minDB)
            else:
                # log scale mapping, why 0.6?!
                volumeDB = 20*np.log10(volumeVal)/0.6
            if volumeDB<self.minDB:
                volumeDB = self.minDB
        else:
            volumeDB = self.minDB

        print(volumeVal,volumeDB) # for debugging
        self.selectedDevice.SetMasterVolumeLevel(volumeDB, None)


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
                        while self.recording:
                            if self.audio_record_queue.not_empty:
                                try:
                                    audio_output = self.audio_record_queue.get(timeout=1)
                                    file.write(audio_output)
                                    time.sleep(1)
                                except self.audio_record_queue.not_empty:
                                    continue
                print('exit thread...')
                
            if not self.thread:
                print("start thread")
                self.thread = threading.Thread(target=recording_thread)
                self.thread.start()
        else:
            print('stopping audio record thread...')
            # stop the thread here
            
            

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




