
from ui.main_window import Ui_MainWindow

from PyQt5 import QtWidgets, QtCore, QtGui
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

import sys


def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / (num_locations + 1e-8)


class StyleContentModel(tf.keras.models.Model):

    """Needs 2.15GiB memory."""

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        # Expects float input in [0,1]
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def style_content_loss(outputs, style_targets, content_targets,
                       style_weight=1e-2, content_weight=1e4):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    num_style_outputs = len(style_outputs)
    num_content_outputs = len(content_outputs)
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_outputs

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_outputs
    loss = style_loss + content_loss
    return loss


def build_model():
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    model = StyleContentModel(style_layers, content_layers)
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
    tensor = tensor * 255.
    array = np.array(tensor, dtype=np.uint8)
    return array


def get_img_ratio(img, max_height=320, channel_first=False):
    ratio = 1
    if isinstance(img, np.ndarray):
        if channel_first:
            ratio = max_height / img.shape[1]
        else:
            ratio = max_height / img.shape[0]
    elif isinstance(img, Image.Image):
        ratio = max_height / img.height
    return ratio


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


class Main(QtWidgets.QMainWindow):

    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Init Model and variables for training
        self.model = build_model()
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        self.content_target = None
        self.style_target = None
        self.content_image_tensor = None
        self.model.trainable = False

        # Init Training timer
        self.apply_grad_timer = QtCore.QTimer()
        self.apply_grad_timer.setInterval(int(1000 / self.ui.spinBox_ExpectedFPS.value()))
        self.apply_grad_timer.timeout.connect(self.on_train_timer_timeout)

        # Init CV Camera
        self.camera = cv2.VideoCapture(0)
        self.camera_capture = None

        # Init Camera timer
        self.camera_timer = QtCore.QTimer()
        self.camera_timer.setInterval(int(1000 / 30))
        self.camera_timer.timeout.connect(self.on_camera_timer_timeout)

        # Signals & Slots
        self.ui.spinBox_ExpectedFPS.valueChanged.connect(self.change_fps)
        self.ui.pushButton_ImportContentPicFile.clicked.connect(self.get_image_file)
        self.ui.pushButton_ImportStylePicFile.clicked.connect(self.get_image_file)
        self.ui.pushButton_TrainModel.clicked.connect(self.toggle_train)
        self.ui.pushButton_TrainTerminate.clicked.connect(self.toggle_train)
        self.ui.pushButton_CameraOpen.clicked.connect(self.toggle_camera)
        self.ui.pushButton_CameraCapture.clicked.connect(self.toggle_camera)
        self.ui.pushButton_DownloadResultPic.clicked.connect(self.download_result)
        self.ui.pushButton_Reset.clicked.connect(self.reset_model_clicked)

    def reset_model(self):
        tf.compat.v1.reset_default_graph()
        self.model = build_model()
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        self.content_target = None
        self.style_target = None
        self.content_image_tensor = None
        self.model.trainable = False

    @QtCore.pyqtSlot()
    def reset_model_clicked(self):
        self.reset_model()
        self.ui.pushButton_DownloadResultPic.setEnabled(False)
        self.ui.pushButton_ImportContentPicFile.setEnabled(True)
        self.ui.pushButton_CameraOpen.setEnabled(True)
        self.ui.pushButton_ImportStylePicFile.setEnabled(True)
        self.ui.label_ResultPic.clear()
        self.ui.label_ContentPic.clear()
        self.ui.label_StylePic.clear()
        self.ui.plainTextEdit_Console.appendPlainText("All reset.")

    @QtCore.pyqtSlot()
    def download_result(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', '.',
                                                             'Images (*.png)')
        if file_name:
            self.ui.label_ResultPic.pixmap().save(file_name, 'PNG')

    @QtCore.pyqtSlot()
    def toggle_camera(self):
        if self.sender() is self.ui.pushButton_CameraOpen:
            self.camera_timer.start()
            self.sender().setEnabled(False)
            self.ui.pushButton_CameraCapture.setEnabled(True)
            self.ui.pushButton_ImportContentPicFile.setEnabled(False)
        elif self.sender() is self.ui.pushButton_CameraCapture:
            self.camera_timer.stop()
            self.sender().setEnabled(False)
            self.ui.pushButton_CameraOpen.setEnabled(True)
            self.ui.pushButton_ImportContentPicFile.setEnabled(True)

            # Reset camera
            self.camera.release()
            self.camera = cv2.VideoCapture(0)
            # Assign variables for training
            self.content_image_tensor = tf.Variable(self.camera_capture[np.newaxis, :, :, ::-1] / 255.)
            self.content_target = self.model(self.camera_capture[np.newaxis, :, :, ::-1] / 255.)['content']
            self.camera_capture = None

    @QtCore.pyqtSlot()
    def on_camera_timer_timeout(self):
        ret, frame = self.camera.read()
        if ret:
            ratio = get_img_ratio(frame)
            frame = cv2.resize(frame, (0, 0), fx=ratio, fy=ratio)
            self.camera_capture = frame[..., ::-1].copy()
            qim = array_uint8_to_qimage(frame, bgr=True)
            self.ui.label_ContentPic.setPixmap(QtGui.QPixmap.fromImage(qim))
        else:
            self.ui.plainTextEdit_Console.appendPlainText(
                "Camera capture failed."
            )
            self.camera_timer.stop()
            self.ui.pushButton_CameraOpen.setEnabled(True)
            self.ui.pushButton_CameraCapture.setEnabled(False)
            self.ui.pushButton_ImportContentPicFile.setEnabled(True)

    @QtCore.pyqtSlot()
    def toggle_train(self):
        if self.sender() is self.ui.pushButton_TrainModel:
            self.apply_grad_timer.start()
            self.ui.pushButton_TrainTerminate.setEnabled(True)
            self.ui.pushButton_ImportContentPicFile.setEnabled(False)
            self.ui.pushButton_CameraOpen.setEnabled(False)
            self.ui.pushButton_ImportStylePicFile.setEnabled(False)
            self.ui.pushButton_Reset.setEnabled(False)
            self.sender().setEnabled(False)
        elif self.sender() is self.ui.pushButton_TrainTerminate:
            self.apply_grad_timer.stop()
            self.ui.pushButton_TrainModel.setEnabled(True)
            self.ui.pushButton_Reset.setEnabled(True)
            self.sender().setEnabled(False)

    @staticmethod
    def train_step(content_img_tensor, style_target, content_target, model, opt):
        with tf.GradientTape() as tape:
            outputs = model(content_img_tensor)
            loss = style_content_loss(outputs, style_target, content_target)
        grad = tape.gradient(loss, content_img_tensor)
        opt.apply_gradients([(grad, content_img_tensor)])
        content_img_tensor.assign(clip_0_1(content_img_tensor))

    @QtCore.pyqtSlot()
    def on_train_timer_timeout(self):
        condition = [
            self.content_image_tensor is not None,
            self.style_target is not None,
            self.content_target is not None
        ]
        if all(condition):
            self.train_step(self.content_image_tensor, self.style_target,
                            self.content_target, self.model, self.opt)
            result_image_array = tensor_to_array(self.content_image_tensor)[0]
            qim = array_uint8_to_qimage(result_image_array)
            self.ui.label_ResultPic.setPixmap(QtGui.QPixmap.fromImage(qim))
            self.ui.pushButton_DownloadResultPic.setEnabled(True)
        else:
            self.ui.plainTextEdit_Console.appendPlainText(
                f"Has style image: {condition[1]}, "
                f"has content image: {condition[2]}."
            )
            self.apply_grad_timer.stop()

    @QtCore.pyqtSlot()
    def get_image_file(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '.',
                                                             "Images (*.png *.jpg)")
        if file_name:
            img: Image.Image = Image.open(file_name)
            ratio = get_img_ratio(img)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)))
            img = np.asarray(img)
            qimg = array_uint8_to_qimage(img, bgr=False)
            if self.sender() is self.ui.pushButton_ImportContentPicFile:
                self.ui.label_ContentPic.setPixmap(QtGui.QPixmap.fromImage(qimg))
                self.content_image_tensor = tf.Variable(img[np.newaxis, :, :, :] / 255.)
                self.content_target = self.model(img[np.newaxis, :, :, :] / 255.)['content']
                self.ui.plainTextEdit_Console.appendPlainText(f"Loaded Content Pic "
                                                              f"from {file_name}")
            elif self.sender() is self.ui.pushButton_ImportStylePicFile:
                self.ui.label_StylePic.setPixmap(QtGui.QPixmap.fromImage(qimg))
                self.style_target = self.model(img[np.newaxis, :, :, :] / 255.)['style']
                self.ui.plainTextEdit_Console.appendPlainText(f"Loaded Style Pic "
                                                              f"from {file_name}")

    @QtCore.pyqtSlot()
    def change_fps(self):
        self.apply_grad_timer.setInterval(int(1000 / self.ui.spinBox_ExpectedFPS.value()))
        self.ui.plainTextEdit_Console.appendPlainText(
            f"Expect FPS changed to {self.ui.spinBox_ExpectedFPS.value()}.")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec())
