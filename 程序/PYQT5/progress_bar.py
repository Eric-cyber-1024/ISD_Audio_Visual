from proglog import ProgressBarLogger
from PyQt5.QtWidgets import QApplication, QProgressDialog
from PyQt5.QtCore import QTimer, QObject, pyqtSignal, pyqtSlot

class MySignalEmitter(QObject):
    # manage signal for non-QObject class
    percentage_changed_signal = pyqtSignal(float)
    text_changed_signal = pyqtSignal(str)

class CustomProgressBarLogger(ProgressBarLogger):

    def __init__(self):
        super().__init__()
        self.prev_percentage = 0.0
        self.signal_emitter = MySignalEmitter()

    # def callback(self, **changes):
        # Every time the logger message is updated, this function is called with
        # the `changes` dictionary of the form `parameter: new value`.
        # for (parameter, value) in changes.items():
        #     print ('Parameter %s is now %s' % (parameter, value))

    def bars_callback(self, bar, attr, value,old_value=None):
        # Every time the logger progress is updated, this function is called        
        percentage = (value / self.bars[bar]['total']) * 100
        
        self.percentage = percentage
        if attr == 'total':
            self.prev_percentage = 0.0
            if bar == 'chunk':
                self.signal_emitter.text_changed_signal.emit("1/2")
            elif bar == 't':
                self.signal_emitter.text_changed_signal.emit("2/2")
        if attr == 'index' and (percentage - self.prev_percentage >= 1.0 or percentage == 100):
            self.signal_emitter.percentage_changed_signal.emit(percentage)
            self.prev_percentage = percentage
            # print(bar,attr,percentage)
            # print(bar)
    
