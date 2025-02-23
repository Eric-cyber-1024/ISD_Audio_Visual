from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from Style import *

BUTTON_FONT = QFont("Arial",40)
Label_Font = QFont("Arial",20)


def Create_Button(Title,function,Style):
        Button= QPushButton()
        Button.clicked.connect(function)
        Button.setFixedSize(NUM_BUTTON_WIDTH,NUM_BUTTON_HEIGHT)
        Button.setText(Title)
        Button.setFont(BUTTON_FONT )
        Button.setStyleSheet(Style)
        return Button



def Create_RadioBotton(Title,function):
        Target = QRadioButton(Title) 
        Target.setGeometry(200, 150, 100, 30) 
        Target.setChecked(False) 
        Target.setFont(BUTTON_FONT)
        Target.setStyleSheet(RADIO_STYLE)
        Target.clicked.connect(function)
        return Target

def Create_Label():
        label=QPlainTextEdit()
        label.setFixedSize(250,800)
        label.setStyleSheet("background-color: white ;")
        label.setFont(Label_Font)
        return label

def Create_ImageWindow():
        frame = QLabel()
        frame.setText("Loading Camera Frame ...")
        frame.setStyleSheet("")
        frame.setFont(QFont("Arial",40))
        frame.setFixedSize(WINDOW_WIDTH,WINDOW_HEIGHT)
        frame.setStyleSheet("background-color:Grey;border-width: 4px;border-radius: 20px;alignment:center")
        frame.setContentsMargins(0,0,0,0)
        return frame

def Create_Slider(Minimum,Maximum,Value,Step,Style,function):
        slider=QSlider(Qt.Horizontal)
        slider.setMinimum(Minimum)
        slider.setMaximum(Maximum)
        slider.setValue(Value)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(Step)
        slider.setFont(QFont("Arial",40))
        slider.setStyleSheet(Style)
        slider.setFixedWidth(1000)
        slider.valueChanged.connect(function)
        return slider