# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.plainTextEdit_Terminal = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_Terminal.setGeometry(QtCore.QRect(0, 400, 800, 152))
        self.plainTextEdit_Terminal.setMinimumSize(QtCore.QSize(800, 152))
        self.plainTextEdit_Terminal.setReadOnly(True)
        self.plainTextEdit_Terminal.setObjectName("plainTextEdit_Terminal")
        self.pushButton_Terminate = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Terminate.setEnabled(False)
        self.pushButton_Terminate.setGeometry(QtCore.QRect(650, 20, 101, 31))
        self.pushButton_Terminate.setObjectName("pushButton_Terminate")
        self.groupBox_Programs = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_Programs.setGeometry(QtCore.QRect(0, 80, 801, 321))
        self.groupBox_Programs.setObjectName("groupBox_Programs")
        self.pushButton_NeuralTransfer = QtWidgets.QPushButton(self.groupBox_Programs)
        self.pushButton_NeuralTransfer.setGeometry(QtCore.QRect(650, 120, 101, 31))
        self.pushButton_NeuralTransfer.setObjectName("pushButton_NeuralTransfer")
        self.pushButton_InsightFace = QtWidgets.QPushButton(self.groupBox_Programs)
        self.pushButton_InsightFace.setGeometry(QtCore.QRect(650, 70, 101, 31))
        self.pushButton_InsightFace.setObjectName("pushButton_InsightFace")
        self.pushButton_MNIST = QtWidgets.QPushButton(self.groupBox_Programs)
        self.pushButton_MNIST.setGeometry(QtCore.QRect(650, 20, 101, 31))
        self.pushButton_MNIST.setObjectName("pushButton_MNIST")
        self.pushButton_DCGAN = QtWidgets.QPushButton(self.groupBox_Programs)
        self.pushButton_DCGAN.setGeometry(QtCore.QRect(650, 170, 101, 31))
        self.pushButton_DCGAN.setObjectName("pushButton_DCGAN")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        self.menuOptions = QtWidgets.QMenu(self.menubar)
        self.menuOptions.setObjectName("menuOptions")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuOptions.addAction(self.actionQuit)
        self.menubar.addAction(self.menuOptions.menuAction())

        self.retranslateUi(MainWindow)
        self.actionQuit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_Terminate.setText(_translate("MainWindow", "Terminate"))
        self.groupBox_Programs.setTitle(_translate("MainWindow", "Programs"))
        self.pushButton_NeuralTransfer.setText(_translate("MainWindow", "NeuralTransfer"))
        self.pushButton_InsightFace.setText(_translate("MainWindow", "InsightFace"))
        self.pushButton_MNIST.setText(_translate("MainWindow", "MNIST"))
        self.pushButton_DCGAN.setText(_translate("MainWindow", "DCGAN"))
        self.menuOptions.setTitle(_translate("MainWindow", "Options"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
