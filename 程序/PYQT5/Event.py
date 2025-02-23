from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from Style import *
from moviepy.editor import VideoFileClip, AudioFileClip
import os 
from Enum_library import  *
# Create neccessary Directory
def check_folder_existence(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"The folder '{folder_path}' exists.")
    else:
        print(f"The folder '{folder_path}' does not exist.")
        os.makedirs(folder_path)


def update_label(value):
        # Update the label text with the current slider value
        print(value)


def start_recording(self):
        self.is_recording = True
        self.text_label.appendPlainText('Status: Recording')
        self.audio_thread.start()

def combine_video_audio(video_path, audio_path, output_path):
    global OUTPUT_NAME,AUDIO_NAME,VIDEO_NAME
    ffmpeg_options = "-analyzeduration 100M -probesize 100M"
    # Load the video clip
    video_clip = VideoFileClip(video_path, audio=False)
    # Cut off the first 3 second to align with the audio ( might need adjustment)
    video_clip = video_clip.subclip(3)
    # Load the audio clip
    audio_clip = AudioFileClip(audio_path)

    # Set the audio of the video to the loaded audio clip
    video_clip = video_clip.set_audio(audio_clip)
    # Write the combined clip to a new file
    video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", temp_audiofile='temp.m4a', remove_temp=False)

def switchPage(self,PAGE):         
        self.stacked_widget.setCurrentIndex(PAGE)



def ToggleSelection(self,choice):
        self.LPF_Select = choice
        i = 0
        list_of_range = [self.CheckBox_6kHz,self.CheckBox_12kHz,self.CheckBox_18kHz,self.CheckBox_Full]
        list_frequency = [Frequency_Selection.LPF_6K,Frequency_Selection.LPF_12K,Frequency_Selection.LPF_18K,Frequency_Selection.LPF_FULL]
        for item in list_frequency:
            if item.value != choice:
                list_of_range[i].setChecked(False)
            else:
                    list_of_range[i].setChecked(True)
            i+=1

def Exit():
        exit()


