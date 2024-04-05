from proglog import ProgressBarLogger
from PyQt5.QtWidgets import QApplication, QProgressDialog
from PyQt5.QtCore import QTimer, QObject, pyqtSignal, pyqtSlot

class MySignalEmitter(QObject):
    # manage signal for non-QObject class
    percentage_changed_signal = pyqtSignal(float)

class CustomProgressBarLogger(ProgressBarLogger):

    def __init__(self):
        super().__init__()
        self.signal_emitter = MySignalEmitter()

    # def callback(self, **changes):
        # Every time the logger message is updated, this function is called with
        # the `changes` dictionary of the form `parameter: new value`.
        # for (parameter, value) in changes.items():
        #     print ('Parameter %s is now %s' % (parameter, value))

    def bars_callback(self, bar, attr, value,old_value=None):
        # Every time the logger progress is updated, this function is called        
        percentage = (value / self.bars[bar]['total']) * 100
        # print(bar,attr,percentage)
        self.percentage = percentage
        if attr == 'index':
            self.signal_emitter.percentage_changed_signal.emit(percentage)
    
