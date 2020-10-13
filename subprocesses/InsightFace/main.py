
from ui.main_window import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui, QtCore

import cv2
import numpy as np
import insightface
from PIL import ImageFont, Image, ImageDraw

from copy import deepcopy
import pickle
import sys
import os

THRESHOLD = 0.9

EXPECTED_FPS = 15

FONT_HEIGHT = 15
LINE_SPACING = 4
CJK_FONT = ImageFont.truetype('NotoSansMonoCJKsc-Regular.otf', size=FONT_HEIGHT)


def cosine_similarity(emb_true, emb_pred) -> float:
    """
    Compute cosine similarity.

    :param emb_true: np.ndarray, a vector.
    :param emb_pred: np.ndarray, same shape as emb_true.
    :return: float, calculated cosine similarity, whose value is between [0, 1].
    """
    epsilon = 1e-8
    result = (np.dot(emb_true, emb_pred) /
              (np.linalg.norm(emb_true) * np.linalg.norm(emb_pred) + epsilon) *
              0.5 + 0.5)
    return result


np_cosine_similarity = np.vectorize(cosine_similarity, otypes=['float32'],
                                    signature='(d),(d)->()')


def mark_faces_from_frame(frame, face, face_names, face_emb_matrix,
                          order=0, threshold=0.8) -> np.ndarray:
    """
    Mark faces on the frame

    :param frame: np.ndarray(3D), the frame of camera.
    :param face: Face object detected by InsightFace.
    :param face_names: Built-in dict enumerates names of face in storage.
    :param face_emb_matrix: np.ndarray(2D), list of face embeddings.
    :param order: Built-in int, the order of faces in frame.
    :param threshold: Built-in float, the least cosine similarity to determine the name.

    :return: np.ndarray(3D), the rewritten frame to show.
    """
    display_name = '知らない人'
    display_confidence = 0
    color = (0, 0, 255)  # In order BGR
    if threshold > 1:
        threshold = 1
    elif threshold < 0:
        threshold = 0

    # Face Bounding Box
    bbox = face.bbox.astype('int32')
    cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color=color)

    # Show recognition
    if len(face_emb_matrix) > 0:
        mapped_similarity = np_cosine_similarity(face_emb_matrix, face.embedding)
        max_idx = mapped_similarity.argmax()
        if mapped_similarity[max_idx] > threshold:
            display_name = face_names[max_idx]
            display_confidence = mapped_similarity.max()
    # Show recognition: Draw Confidence
    confidence_bar_height = int(bbox[3] + display_confidence * (bbox[1] - bbox[3]))
    confidence_bar_width = 10
    cv2.rectangle(frame, tuple(bbox[2:]),
                  (bbox[2] + confidence_bar_width, confidence_bar_height),
                  color=color, thickness=-1)
    # Show recognition: Show Identity
    frame = Image.fromarray(frame)
    drawer = ImageDraw.Draw(frame)
    drawer.text((bbox[0], bbox[1] - FONT_HEIGHT - LINE_SPACING),
                display_name, fill=color, font=CJK_FONT, spacing=LINE_SPACING)
    drawer.multiline_text((bbox[0], bbox[3]), f"Age: {face.age}\nOrder: {order}",
                          fill=color, font=CJK_FONT, spacing=LINE_SPACING)
    frame = np.array(frame)

    return frame


def array_uint8_rgb_to_qimage(array: np.ndarray) -> QtGui.QImage:
    qim = QtGui.QImage(array.data,
                       array.shape[1],
                       array.shape[0],
                       array.strides[0],
                       QtGui.QImage.Format_RGB888).rgbSwapped()
    return qim


class Main(QtWidgets.QMainWindow):

    face_embeddings_filename = 'embeddings.pkl'

    def __init__(self):
        super(Main, self).__init__()
        # Init UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Init Camera by OpenCV
        self.vid_capture = cv2.VideoCapture(0)
        self.camera_frame = None
        self.captured_frame = None

        # Init InsightFace models
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(0)  # Use GPU, int value < 0 would uses CPU.
        # Init embeddings storage
        self.face_embeddings = {}
        if not os.path.exists(self.face_embeddings_filename):
            with open(self.face_embeddings_filename, 'wb') as f:
                pickle.dump(self.face_embeddings, f)
        else:
            with open(self.face_embeddings_filename, 'rb') as f:
                self.face_embeddings = pickle.load(f)
        self.face_names = {i: k for (i, k) in enumerate(self.face_embeddings.keys())}
        self.face_emb_matrix = np.array(list(self.face_embeddings.values()))
        self.camera_faces = []
        self.captured_faces = []

        # The camera capture loop
        self.camera_timer = QtCore.QTimer()
        self.camera_timer.setInterval(int(1000 / EXPECTED_FPS))
        self.camera_timer.timeout.connect(self.predict_image)

        # Signals & Slots
        self.ui.pushButton_CameraStop.setDisabled(True)
        self.ui.pushButton_CameraOpen.clicked.connect(self.toggle_camera)
        self.ui.pushButton_CameraStop.clicked.connect(self.toggle_camera)
        self.ui.pushButton_CameraCapture.clicked.connect(self.capture_frame)
        self.ui.pushButton_NameSubmit.clicked.connect(self.submit_faces)

    def update_embeddings(self, name, embed):
        self.face_embeddings[name] = embed

        self.face_names = {i: k for (i, k) in enumerate(self.face_embeddings.keys())}
        self.face_emb_matrix = np.array(list(self.face_embeddings.values()))
        # Store Embeddings
        with open('embeddings.pkl', 'wb') as f:
            pickle.dump(self.face_embeddings, f)

    @QtCore.pyqtSlot()
    def submit_faces(self):
        line_edit_group = list(sorted(self.ui.groupBox_NameOfFace.findChildren(QtWidgets.QLineEdit),
                                      key=lambda w: w.objectName()))
        texts_in_line_edit_group = [w.text() for w in line_edit_group if w.isEnabled()]
        embeddings = [f.embedding for f in self.captured_faces]
        for k, v in zip(texts_in_line_edit_group, embeddings):
            k = k.strip()
            if k:
                similarity_mat = np_cosine_similarity(self.face_emb_matrix, v)
                similarity_max = similarity_mat.max()
                similarity_argmax = similarity_mat.argmax()
                if k not in self.face_embeddings:
                    if similarity_max < THRESHOLD:
                        self.update_embeddings(k, v)
                    else:
                        self.ui.plainTextEdit_Console.appendPlainText(
                            f"{self.face_names[similarity_argmax]} in saved embeddings has "
                            f"similarity {similarity_max} higher than threshold {THRESHOLD}, "
                            f"embedding will not be saved."
                        )
                else:
                    if similarity_max > THRESHOLD:
                        self.ui.plainTextEdit_Console.appendPlainText(
                            f"{repr(k)} in saved embeddings, will be overwritten."
                        )
                        self.update_embeddings(k, v)
                    else:
                        self.ui.plainTextEdit_Console.appendPlainText(
                            f"{repr(k)} in saved embeddings, but similarity is lower than {THRESHOLD}, "
                            f"will not be overwritten."
                        )

    @QtCore.pyqtSlot()
    def capture_frame(self):
        if self.camera_frame is not None and self.camera_timer.isActive():
            self.camera_timer.stop()
            self.camera_timer.singleShot(1, self.predict_image)
            self.captured_frame = self.camera_frame.copy()
            self.captured_faces = deepcopy(self.camera_faces)
            qim = array_uint8_rgb_to_qimage(self.captured_frame)
            qpix = QtGui.QPixmap.fromImage(qim)
            self.ui.label_Capture.setPixmap(qpix)
            line_edit_group = list(sorted(self.ui.groupBox_NameOfFace.findChildren(QtWidgets.QLineEdit),
                                          key=lambda w: w.objectName()))
            for i, widget in enumerate(line_edit_group):
                widget.clear()
                if i < len(self.captured_faces):
                    widget.setEnabled(True)
                else:
                    widget.setEnabled(False)
            self.camera_timer.start()

    @QtCore.pyqtSlot()
    def predict_image(self):
        ret, frame = self.vid_capture.read()
        if ret:
            self.camera_faces = self.model.get(frame[..., ::-1])
            for i, face in enumerate(self.camera_faces):
                frame = mark_faces_from_frame(
                    frame, face,
                    self.face_names, self.face_emb_matrix, order=i, threshold=THRESHOLD)
            self.camera_frame = frame.copy()
            qim = array_uint8_rgb_to_qimage(self.camera_frame)
            qpix = QtGui.QPixmap.fromImage(qim)
            self.ui.label_Camera.setPixmap(qpix)
        else:
            self.ui.plainTextEdit_Console.appendPlainText("Camera failed to capture frame.")
            self.ui.pushButton_CameraStop.click()

    @QtCore.pyqtSlot()
    def toggle_camera(self):
        if self.sender() is self.ui.pushButton_CameraOpen:
            self.camera_timer.start()
            self.sender().setDisabled(True)
            self.ui.pushButton_CameraStop.setEnabled(True)
        elif self.sender() is self.ui.pushButton_CameraStop:
            self.camera_timer.stop()
            self.sender().setDisabled(True)
            self.ui.pushButton_CameraOpen.setEnabled(True)
            self.camera_frame = None


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec())
