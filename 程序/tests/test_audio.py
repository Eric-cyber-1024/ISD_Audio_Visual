from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


from pycaw.pycaw import AudioUtilities

from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox
from PyQt5.QtMultimedia import QAudioDeviceInfo, QAudio, QAudioInput, QAudioFormat
from PyQt5.QtWidgets import QApplication, QComboBox, QDialog, QDialogButtonBox, QVBoxLayout, QPushButton

import time
import sounddevice as sd
import soundfile as sf
from threading import Thread

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

        # Create start recording button
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.start_recording)
        layout.addWidget(self.record_button)

        # Create dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
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
                with sd.InputStream(device=input_device.split(',')[1], channels=2, callback=audio_callback):
                    # Wait for ten seconds or until recording is stopped
                    time.sleep(10)
                # Save the recorded audio as a WAV file
                if self.frames:
                    filename = "recorded_audio.wav"
                    with sf.SoundFile(filename, mode='w', samplerate=44100,channels=2) as file:
                        for frame in self.frames:
                            file.write(frame)
                        print(f"Recording saved as {filename}")
                else:
                    print("No audio recorded.")
                # Reset recording variables
                self.recording = False
                self.frames = []

            # Start the recording thread
            self.thread = Thread(target=recording_thread)
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
    

def devToStr(device):
    '''
    get info of a sounddevice device
    '''
    hostapi_names = [hostapi['name'] for hostapi in sd.query_hostapis()]
    text = u'{name}, {ha} ({ins} in, {outs} out)'.format(
            name=device['name'],
            ha=hostapi_names[device['hostapi']],
            ins=device['max_input_channels'],
            outs=device['max_output_channels'])
    return text

def show_audio_device_dialog_sd():
    # Get the available audio input and output devices
    devices = sd.query_devices()
    input_devices  = ['%d,%s' %(i,devToStr(device)) for i,device in enumerate(devices) if device['max_input_channels']  > 0]
    output_devices = ['%d,%s' %(i,devToStr(device)) for i,device in enumerate(devices) if device['max_output_channels'] > 0]

    # Create and show the audio device dialog
    dialog = AudioDeviceDialog(input_devices, output_devices)
    if dialog.exec_() == QDialog.Accepted:
        input_device, output_device = dialog.get_selected_devices()
        # Process the selected audio devices
        print(f"Selected Audio Input Device: {input_device}")
        print(f"Selected Audio Output Device: {output_device}")


def show_audio_input_dialog():
    # Get the available audio input devices
    audio_device_info = QAudioDeviceInfo.defaultInputDevice()
    available_devices = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)

    # Create a list of device names
    device_names = [device.deviceName() for device in available_devices]

    # Show the input dialog to select the audio input device
    selected_device, ok = QInputDialog.getItem(None, "Select Audio Input Device", "Input Device:", device_names, 0, False)

    if ok and selected_device:
        # Find the selected device by name
        selected_device_info = None
        for device in available_devices:
            if device.deviceName() == selected_device:
                selected_device_info = device
                break

        if selected_device_info:
            # Use the selected device for audio input
            audio_format = QAudioFormat()
            audio_format.setSampleRate(44100)
            audio_format.setChannelCount(1)
            audio_format.setSampleSize(16)
            audio_format.setCodec("audio/pcm")
            audio_format.setByteOrder(QAudioFormat.LittleEndian)
            audio_format.setSampleType(QAudioFormat.SignedInt)

            audio_input = QAudioInput(selected_device_info, audio_format)

            # Show a message box with the selected device information
            message = f"Selected Audio Input Device:\n\n{selected_device_info.deviceName()}\n{selected_device_info.deviceDescription()}"
            QMessageBox.information(None, "Audio Input Device Selected", message)
        else:
            QMessageBox.warning(None, "Device Not Found", "Selected audio input device not found.")
    else:
        QMessageBox.warning(None, "No Device Selected", "No audio input device selected.")





class AudioController:
    def __init__(self, process_name):
        self.process_name = process_name
        self.volume = self.process_volume()

    def listAllSections(self):
        sessions = AudioUtilities.GetAllSessions()
        print('number of sessions=%d' %(len(sessions)))
        for session in sessions:
            interface = session.SimpleAudioVolume
            print(session,interface)


    def mute(self):
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            interface = session.SimpleAudioVolume
            if session.Process and session.Process.name() == self.process_name:
                interface.SetMute(1, None)
                print(self.process_name, "has been muted.")  # debug

    def unmute(self):
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            interface = session.SimpleAudioVolume
            if session.Process and session.Process.name() == self.process_name:
                interface.SetMute(0, None)
                print(self.process_name, "has been unmuted.")  # debug

    def process_volume(self):
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            interface = session.SimpleAudioVolume
            if session.Process and session.Process.name() == self.process_name:
                print("Volume:", interface.GetMasterVolume())  # debug
                return interface.GetMasterVolume()

    def set_volume(self, decibels):
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            interface = session.SimpleAudioVolume
            if session.Process and session.Process.name() == self.process_name:
                # only set volume in the range 0.0 to 1.0
                self.volume = min(1.0, max(0.0, decibels))
                interface.SetMasterVolume(self.volume, None)
                print("Volume set to", self.volume)  # debug

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




if __name__ == "__main__":
    

    audio_controller = AudioController("chrome.exe")
    audio_controller.listAllSections()

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    
    # Control volume
    #volume.SetMasterVolumeLevel(-0.0, None) #max
    #volume.SetMasterVolumeLevel(-5.0, None) #72%
    volume.SetMasterVolumeLevel(-20.0, None) #51%

    app = QApplication([])
    show_audio_device_dialog_sd()
    app.exec_()





