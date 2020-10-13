
from ui.main_window import Ui_MainWindow
from PyQt5 import QtWidgets, QtCore

import sys
import os

FILE = os.path.abspath(__file__)  # Record this variable for run.ps1


class Main(QtWidgets.QMainWindow):

    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.process = QtCore.QProcess()
        self.process.stateChanged.connect(self.onStateChanged)
        self.process.readyReadStandardError.connect(self.terminalAppendError)
        self.process.readyReadStandardOutput.connect(self.terminalAppendOutput)
        self.process.finished.connect(self.onFinished)

        self.ui.pushButton_MNIST.clicked.connect(self.openSubprocess)
        self.ui.pushButton_InsightFace.clicked.connect(self.openSubprocess)
        self.ui.pushButton_NeuralTransfer.clicked.connect(self.openSubprocess)
        self.ui.pushButton_DCGAN.clicked.connect(self.openSubprocess)
        self.ui.pushButton_Terminate.clicked.connect(self.terminateSubprocess)

    @QtCore.pyqtSlot()
    def onFinished(self):
        os.chdir(os.path.abspath(os.path.dirname(FILE)))
        self.ui.pushButton_Terminate.setEnabled(False)
        for w in self.ui.groupBox_Programs.findChildren(QtWidgets.QPushButton):
            w.setEnabled(True)

    @QtCore.pyqtSlot()
    def terminalAppendError(self):
        error = self.process.readAllStandardError().data().decode()
        self.ui.plainTextEdit_Terminal.appendPlainText(error)

    @QtCore.pyqtSlot()
    def terminalAppendOutput(self):
        output = self.process.readAllStandardOutput().data().decode()
        self.ui.plainTextEdit_Terminal.appendPlainText(output)

    @QtCore.pyqtSlot()
    def terminateSubprocess(self):
        self.process.kill()
        self.ui.plainTextEdit_Terminal.appendPlainText(
            "You terminated sub program."
        )

    @QtCore.pyqtSlot()
    def openSubprocess(self):
        if self.sender() is self.ui.pushButton_InsightFace:
            os.chdir('./subprocesses/InsightFace')
            self.process.start('python main.py')
        elif self.sender() is self.ui.pushButton_NeuralTransfer:
            os.chdir('./subprocesses/StyleTransfer')
            self.process.start('python main.py')
        elif self.sender() is self.ui.pushButton_MNIST:
            os.chdir('./subprocesses/MNIST')
            self.process.start('python main.py')
        elif self.sender() is self.ui.pushButton_DCGAN:
            os.chdir('./subprocesses/DCGAN')
            self.process.start('python main.py')
        self.ui.pushButton_Terminate.setEnabled(True)
        for w in self.ui.groupBox_Programs.findChildren(QtWidgets.QPushButton):
            w.setEnabled(False)

    @QtCore.pyqtSlot()
    def onStateChanged(self):
        if (self.process.state()) == 2:
            self.ui.plainTextEdit_Terminal.appendPlainText(
                f'Subprocess starting, PID: {self.process.processId()}.')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec())
