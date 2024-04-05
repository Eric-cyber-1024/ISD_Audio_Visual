WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
# FRAME_WIDTH = 1080
# FRAME_HEIGHT = 810
# FRAME_WIDTH = 640
# FRAME_HEIGHT = 480
FRAME_WIDTH = 1600
FRAME_HEIGHT = 800
BUTTON_BAR_HEIGHT = 200


BUTTON_STYLE_TEXT = """
        QPushButton {
            background: qradialgradient(
            cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
            radius: 1.35, stop: 0 #fff, stop: 1 #888
            );    
            border: 4px solid #000000;
            padding: 10px;
            border-radius: 20px;
        }
        QPushButton:hover {
            color: white;
            background-color: darkgreen
        }
    """ 
BUTTON_STYLE_RED = """
        QPushButton {
            background-color: Gray; 
            color : white;
            border-width: 4px;
            border-radius: 20px;
        }
        QPushButton:hover {
            color: white;
            background-color: red
        }
    """ 
BUTTON_STYLE_SETTING = """
        QPushButton {
            color: #333;
            border: 0px solid #555;
            border-radius: 40px;
            border-style: outset;
            background: qradialgradient(
                cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                radius: 1.35, stop: 0 #fff, stop: 1 #888
                );
            padding: 5px;
            }
        
        QPushButton:hover {
            background: qradialgradient(
                cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                radius: 1.35, stop: 0 #fff, stop: 1 #bbb
                );
            }
        
        QPushButton:pressed {
            border-style: inset;
            background: qradialgradient(
                cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,
                radius: 1.35, stop: 0 #fff, stop: 1 #ddd
                );
            }
    """

BUTTON_STYLE_RECORD = """
        QPushButton {
            color: #333;
            border: 0px solid #555;
            border-radius: 40px;
            border-style: outset;
            background: qradialgradient(
                cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                radius: 1.35, stop: 0 #fff, stop: 1 #888
                );
            padding: 5px;
            }
        
        QPushButton:hover {
            background: qradialgradient(
                cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                radius: 1.35, stop: 0 #fff, stop: 1 #bbb
                );
            }
        
        QPushButton:pressed {
            border-style: inset;
            background: qradialgradient(
                cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,
                radius: 1.35, stop: 0 #fff, stop: 1 #ddd
                );
            }
    """
BUTTON_STYLE_MIC = """
        QPushButton {
            color: #333;
            border: 0px solid #555;
            border-radius: 40px;
            border-style: outset;
            background: qradialgradient(
                cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                radius: 1.35, stop: 0 #fff, stop: 1 #888
                );
            padding: 5px;
            }
        
        QPushButton:hover {
            background: qradialgradient(
                cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                radius: 1.35, stop: 0 #fff, stop: 1 #bbb
                );
            }
        
        QPushButton:pressed {
            border-style: inset;
            background: qradialgradient(
                cx: 0.4, cy: -0.1, fx: 0.4, fy: -0.1,
                radius: 1.35, stop: 0 #fff, stop: 1 #ddd
                );
            }
    """
SLIDER_STYLE = """
            QSlider::groove:horizontal {
                height: 10px;
                border: 1px solid #bbb;
                background: white;
                margin: 0px;
            }

            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #45ADED, stop:1 #00579A);
                border: 1px solid #5c5c5c;
                width: 15px;
                margin: -2px 0;
                border-radius: 3px;
            }

            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #B1CC00, stop:1 #7D8B00);
                height: 10px;
            }

            QSlider::add-page:horizontal {
                background: #eee;
                height: 10px;
            }

            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #78AADB, stop:1 #00579A);
                border: 1px solid #00579A;
            }

            QSlider::sub-page:horizontal:disabled {
                background: #bbb;
                border-color: #999;
            }

            QSlider::add-page:horizontal:disabled {
                background: #eee;
                border-color: #999;
            }

            QSlider::handle:horizontal:disabled {
                background: #eee;
                border: 1px solid #aaa;
            }

            QSlider::tick:horizontal {
                height: 0px;
            }
            """


SLIDER_STYLE_2 = """
                .QSlider {
                    min-height: 38px;
                    max-height: 68px;
               
                }

                .QSlider::groove:horizontal {
                 
                    height: 10px;
                    background: white;
                    
                }

                .QSlider::handle:horizontal {
                    background: darkblue;
                    width: 15px;
                    height: 50px;
                    margin: -24px -0px;
                  
                }

                .QSlider::sub-page:horizontal {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1d45a1, stop:1 #1d45a1);
                    height: 10px;
                }

                .QSlider::add-page:horizontal {
                    background: #eee;
                    height: 10px;
                }

                
                .QSlider::groove:horizontal:disabled {
                    background-color: lightgray;
                    
                }

                .QSlider::handle:horizontal:disabled {
                    background-color: #555;
                }

                .QSlider::sub-page:horizontal:disabled {
                    background: #aaa;
                    height: 10px;
                }

                .QSlider::add-page:horizontal:disabled {
                    background: #aaa;
                    height: 10px;
                }
        """

RADIO_STYLE = """
            QRadioButton {
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 30px;
                height: 30px;
            }
        """

NUM_BUTTON_WIDTH = 200
NUM_BUTTON_HEIGHT = 100
NUM_ROUND_BUTTON_SIZE = 80