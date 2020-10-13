
from ui.main_window import Ui_MainWindow
from PyQt5 import QtCore, QtWidgets, QtGui

import tensorflow as tf
from tensorflow.keras import layers
import cv2
import numpy as np

import sys


def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 512, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 512)),

        layers.Conv2DTranspose(384, 5, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(256, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(192, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(96, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, 5, strides=1, padding='same', use_bias=False,
                               activation='tanh'),
    ])
    return model


def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding='same', input_shape=(224, 224, 3)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(96, 5, strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, 5, strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(192, 5, strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(256, 5, strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(512),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    return model


def array_uint8_to_qimage(array: np.ndarray, bgr=False) -> QtGui.QImage:
    qim = QtGui.QImage(array.data,
                       array.shape[1],
                       array.shape[0],
                       array.strides[0],
                       QtGui.QImage.Format_RGB888)
    if bgr:
        qim = qim.rgbSwapped()
    return qim


def tensor_to_array(tensor):
    tensor = tensor * 127.5 + 127.5
    array = np.array(tensor, dtype=np.uint8)
    return array


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def disc_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def gen_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def train_step(images, gen: tf.keras.Model, disc, gen_opt, disc_opt):
    noise = tf.random.normal([1, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(noise, training=True)

        real_output = disc(images, training=True)
        fake_output = disc(generated_images, training=True)

        genLoss = gen_loss(fake_output)
        discLoss = disc_loss(real_output, fake_output)

    grad_of_gen = gen_tape.gradient(genLoss, gen.trainable_variables)
    grad_of_disc = disc_tape.gradient(discLoss, disc.trainable_variables)

    gen_opt.apply_gradients(zip(grad_of_gen, gen.trainable_variables))
    disc_opt.apply_gradients(zip(grad_of_disc, disc.trainable_variables))

    return generated_images, fake_output


class Main(QtWidgets.QMainWindow):

    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Init models
        self.gen_model = build_generator()
        self.disc_model = build_discriminator()
        self.gen_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.disc_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.target_tensor = None

        # Init camera
        self.camera = cv2.VideoCapture(0)
        self.camera_frame = None

        # Train Timer
        self.trn_timer = QtCore.QTimer()
        self.trn_timer.setInterval(int(1000 / self.ui.spinBox_ExpectedFPS.value()))
        self.trn_timer.timeout.connect(self.do_train_step)

        # Camera Timer
        self.camera_timer = QtCore.QTimer()
        self.camera_timer.setInterval(int(1000 / 24))
        self.camera_timer.timeout.connect(self.read_camera)

        # Signals & Slots
        self.ui.pushButton_Reset.clicked.connect(self.reset_models)
        self.ui.spinBox_ExpectedFPS.valueChanged.connect(self.change_interval)
        self.ui.pushButton_OpenCamera.clicked.connect(self.toggle_camera)
        self.ui.pushButton_CameraCapture.clicked.connect(self.toggle_camera)
        self.ui.pushButton_TrainStart.clicked.connect(self.toggle_train)
        self.ui.pushButton_TrainStop.clicked.connect(self.toggle_train)
        self.ui.pushButton_SaveResult.clicked.connect(self.save_result_img)

    @QtCore.pyqtSlot()
    def save_result_img(self):
        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, caption="Save File", directory=".", filter="Image (*.png)")
        if file_path:
            self.ui.label_Generated.pixmap().save(file_path, format="PNG")

    @QtCore.pyqtSlot()
    def do_train_step(self):
        if self.target_tensor is not None:
            generated_img, disc_output = train_step(self.target_tensor[tf.newaxis, ...],
                                                    self.gen_model, self.disc_model,
                                                    self.gen_opt, self.disc_opt)
            img_array = tensor_to_array(generated_img)[0]
            qim = array_uint8_to_qimage(img_array, bgr=False)
            pix = QtGui.QPixmap.fromImage(qim)
            self.ui.label_Generated.setPixmap(pix)
            self.ui.pushButton_SaveResult.setEnabled(True)
            disc_pred = ("<p style='color: red;'>Fake</p>"
                         if (disc_output[0, 0] < 0).numpy()
                         else "<p style='color: blue;'>Real</p>")
            self.ui.textBrowser_Console.append(disc_pred)
        else:
            self.ui.textBrowser_Console.append("Target Photo hasn't been taken yet.")
            self.ui.pushButton_TrainStop.click()

    @QtCore.pyqtSlot()
    def toggle_train(self):
        if self.sender() is self.ui.pushButton_TrainStart:
            self.trn_timer.start()
            self.ui.pushButton_TrainStop.setEnabled(True)
            self.ui.pushButton_Reset.setEnabled(False)
            self.ui.pushButton_OpenCamera.setEnabled(False)
            self.sender().setEnabled(False)
        elif self.sender() is self.ui.pushButton_TrainStop:
            self.trn_timer.stop()
            self.ui.pushButton_TrainStart.setEnabled(True)
            self.ui.pushButton_Reset.setEnabled(True)
            self.sender().setEnabled(False)

    @QtCore.pyqtSlot()
    def read_camera(self):
        ret, self.camera_frame = self.camera.read()
        if ret:
            # Take lr 208, 432, tb 128, 352
            self.camera_frame = cv2.resize(self.camera_frame, (640, 480))
            cv2.rectangle(self.camera_frame, (207, 127), (432, 352), (0, 0, 255))
            qim = array_uint8_to_qimage(self.camera_frame, bgr=True)
            pix = QtGui.QPixmap.fromImage(qim)
            self.ui.label_Camera.setPixmap(pix)
        else:
            self.ui.textBrowser_Console.append("<p style='color: #FF0000'>"
                                               "Camera capture failed."
                                               "</p>")
            self.camera_timer.stop()

    @QtCore.pyqtSlot()
    def toggle_camera(self):
        if self.sender() is self.ui.pushButton_OpenCamera:
            self.camera_timer.start()
            self.ui.pushButton_CameraCapture.setEnabled(True)
            self.ui.pushButton_TrainStart.setEnabled(False)
            self.sender().setEnabled(False)
        elif self.sender() is self.ui.pushButton_CameraCapture:
            self.camera_timer.stop()
            captured_frame = self.camera_frame[128:352, 208:432, ::-1].copy()

            qim = array_uint8_to_qimage(captured_frame, bgr=False)
            pix = QtGui.QPixmap.fromImage(qim)
            self.ui.label_Target.setPixmap(pix)

            self.target_tensor = tf.constant(
                (captured_frame.astype('float32') - 127.5) / 127.5
            )
            self.ui.pushButton_OpenCamera.setEnabled(True)
            self.ui.pushButton_TrainStart.setEnabled(True)
            self.sender().setEnabled(False)

    @QtCore.pyqtSlot()
    def change_interval(self):
        if self.sender() is self.ui.spinBox_ExpectedFPS:
            self.trn_timer.setInterval(int(1000 / self.sender().value()))
            self.ui.textBrowser_Console.append(
                f"Expected FPS Changed to {self.sender().value()}")

    @QtCore.pyqtSlot()
    def reset_models(self):
        tf.compat.v1.reset_default_graph()
        self.gen_model = build_generator()
        self.disc_model = build_discriminator()

        self.ui.label_Camera.clear()
        self.ui.label_Target.clear()
        self.ui.label_Generated.clear()

        self.ui.pushButton_OpenCamera.setEnabled(True)
        self.ui.pushButton_SaveResult.setEnabled(False)

        self.ui.textBrowser_Console.append(f"Models have been reset.")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec())
