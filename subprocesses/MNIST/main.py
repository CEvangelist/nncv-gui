
from ui.main_window import Ui_MainWindow
from PyQt5 import QtWidgets, QtCore, QtGui

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np

import sys


def convert_dtype(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255.
    labels = tf.cast(labels, tf.int32)
    return images, labels


def data_augmentation(images, labels,
                      random_brightness=False,
                      random_contract=False,
                      random_crop=False,
                      random_shift=False):
    images, labels = convert_dtype(images, labels)  # Cast & Scale
    if random_brightness and np.random.random() < 0.5:
        images = tf.image.random_brightness(images, max_delta=0.2)
    if random_contract and np.random.random() < 0.5:
        images = tf.image.random_contrast(images, 0.1, 0.3)
    if random_crop and np.random.random() < 0.5:
        images = tf.image.resize(images, (32, 32))
        images = tf.image.random_crop(images, (28, 28, 1))
    if random_shift and np.random.random() < 0.5:
        images = tf.image.resize_with_crop_or_pad(images, 40, 40)
        images = tf.image.random_crop(images, (28, 28, 1))
    return images, labels


def init_dataset():
    (ds_train, ds_test) = tfds.load('mnist', split=['train', 'test'],
                                    as_supervised=True,
                                    with_info=False)
    return ds_train, ds_test


def build_model():
    from tensorflow.keras import layers
    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool2D(2, 2),
        layers.Flatten(),
        layers.Dense(128),
        layers.Dropout(0.1),
        layers.Dense(10, activation='softmax')
    ])
    return model


def compile_model(model: tf.keras.Sequential, lr=0.001, optim='sgd') -> None:
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    if optim == 'sgd':
        model.compile(
            tf.keras.optimizers.SGD(learning_rate=lr),
            loss=loss,
            metrics=metrics
        )
    elif optim == 'adam':
        model.compile(
            tf.keras.optimizers.Adam(learning_rate=lr),
            loss=loss,
            metrics=metrics
        )
    elif optim == 'rmsprop':
        model.compile(
            tf.keras.optimizers.RMSprop(learning_rate=lr),
            loss=loss,
            metrics=metrics
        )
    else:
        raise ValueError("Parameter `optim` accepts {'sgd', 'adam', 'rmsprop'}, "
                         f"got {optim}")


class ModelThread(QtCore.QThread):
    
    def __init__(self, model, train_ds, test_ds, epochs, callbacks, parent=None):
        super(ModelThread, self).__init__(parent=parent)
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.epochs = epochs
        self.callbacks = callbacks

    def run(self):
        self.model.fit(self.train_ds, epochs=self.epochs,
                       validation_data=self.test_ds,
                       callbacks=self.callbacks)


class Canvas(QtWidgets.QLabel):

    def __init__(self, geometry, parent=None):
        super(Canvas, self).__init__(parent=parent)
        self.setGeometry(geometry)
        self.setToolTip("<p style='font-size: 13pt;'>Canvas: You can draw single "
                        "number on it by pressing mouse key.</p>")
        self.setPixmap(QtGui.QPixmap(self.size()))
        self.pixmap().fill(QtGui.QColor('#FFFFFF'))

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        painter = QtGui.QPainter(self.pixmap())
        pen = painter.pen()
        pen.setWidth(12)
        pen.setColor(QtGui.QColor('#000000'))
        painter.setPen(pen)
        painter.drawPoint(ev.x(), ev.y())
        painter.end()
        self.update()


class Main(QtWidgets.QMainWindow):

    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Init Canvas
        self.label_Canvas = Canvas(self.ui.frame_Canvas.geometry(), self.ui.centralwidget)

        # Init model and dataset
        self.model = build_model()
        self.ds_trn, self.ds_test = init_dataset()

        # Signals & Slots
        self.ui.pushButton_TrainReset.clicked.connect(self.reset_model)
        self.ui.pushButton_TrainStart.clicked.connect(self.start_train)
        self.ui.pushButton_ClearCanvas.clicked.connect(self.clear_canvas)
        self.ui.pushButton_Predict.clicked.connect(self.predict_canvas)
        self.ui.textBrowser_Console.textChanged.connect(self.refresh_console)

    @QtCore.pyqtSlot()
    def refresh_console(self):
        self.ui.textBrowser_Console.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)
        self.ui.textBrowser_Console.update()

    @QtCore.pyqtSlot()
    def predict_canvas(self):
        qim = self.label_Canvas.pixmap().toImage()
        qim = qim.convertToFormat(QtGui.QImage.Format_Grayscale8)
        ba = QtCore.QByteArray()
        buffer = QtCore.QBuffer(ba)
        buffer.open(QtCore.QIODevice.WriteOnly)
        qim.save(buffer, "PNG")
        buffer.close()
        tim = tf.image.decode_png(ba.data(), dtype=tf.uint8)
        tim = tf.cast(tim, tf.float32) / 255.
        tim = tf.ones_like(tim, dtype=tf.float32) - tim
        tim = tf.image.resize(tim, (28, 28))
        pred = self.model.predict(tim[tf.newaxis, ...])
        for i, v in enumerate(pred[0]):
            item = QtWidgets.QTableWidgetItem()
            item.setText(f'{v:.4f}')
            self.ui.tableWidget_PredPosibiity.setItem(0, i, item)

    @QtCore.pyqtSlot()
    def clear_canvas(self):
        self.label_Canvas.pixmap().fill(QtGui.QColor('#FFFFFF'))
        self.label_Canvas.update()

    @QtCore.pyqtSlot()
    def reset_model(self):
        tf.compat.v1.reset_default_graph()
        self.model = build_model()
        self.ds_trn, self.ds_test = init_dataset()

    @QtCore.pyqtSlot()
    def start_train(self):
        ds_trn = (
            self.ds_trn.shuffle(10000).map(
                lambda x, y: data_augmentation(
                    x, y,
                    self.ui.checkBox_RandomBrightness.isChecked(),
                    self.ui.checkBox_RandomContrast.isChecked(),
                    self.ui.checkBox_RandomCrop.isChecked(),
                    self.ui.checkBox_RandomShift.isChecked()
                )
            ).batch(128)
        )
        ds_test = self.ds_test.map(convert_dtype).batch(128)
        compile_model(self.model, self.ui.doubleSpinBox_LearningRate.value(),
                      self.ui.comboBox_Optimizers.currentText().lower())
        callbacks = [tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs:
                self.ui.textBrowser_Console.append(
                    f"Epoch: {epoch+1}, logs: {logs}"
                ))]
        if self.ui.checkBox_EarlyStopping.isChecked():
            callbacks.append(tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True))

        thread = ModelThread(self.model, ds_trn, ds_test, self.ui.spinBox_Epochs.value(),
                             callbacks, self)
        thread.setPriority(QtCore.QThread.LowPriority)
        thread.started.connect(self.toggle_buttons)
        thread.finished.connect(self.toggle_buttons)
        thread.start()

    @QtCore.pyqtSlot()
    def toggle_buttons(self):
        if self.sender().isRunning():
            self.ui.pushButton_TrainReset.setEnabled(False)
            self.ui.pushButton_TrainStart.setEnabled(False)
            self.ui.pushButton_Predict.setEnabled(False)
            self.ui.textBrowser_Console.append("Start training.")
        elif self.sender().isFinished():
            self.ui.pushButton_TrainReset.setEnabled(True)
            self.ui.pushButton_TrainStart.setEnabled(True)
            self.ui.pushButton_Predict.setEnabled(True)
            self.ui.textBrowser_Console.append("Finished training.")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec())
