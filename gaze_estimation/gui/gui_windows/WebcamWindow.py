from PyQt5.QtWidgets import (
    QMainWindow,
    QAction,
    QApplication,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5 import QtGui
import sys
from PyQt5.QtCore import pyqtSlot, Qt, pyqtSignal
import sys
import random
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QColor, QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread
import threading
from abc import ABC, abstractmethod
from gui.gaze_predictor import GazePredictor
import numpy as np
from typing import Tuple
import cv2
import matplotlib.pyplot as plt
from gui.camera import Camera
from gui.gui_windows.ProducerWorker import ProducerWorker
from demo.visualize import add_gaze_to_image


class WebcamWindow(QMainWindow):
    newCameraFrameSignal = pyqtSignal(np.ndarray)
    newGazeVectorSignal = pyqtSignal(tuple)
    windowClosedSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._camera_frame = None
        self._gaze_vector = None
        self.newCameraFrameSignal.connect(self.setCameraFrame)
        self.newGazeVectorSignal.connect(self.setGazeVector)

        self.startCameraFrameProducer()
        self.startGazeVectorProducer()

        self.setWindowTitle("Webcam")
        self.showMaximized()

    def startCameraFrameProducer(self):
        self._camera_frame_producer_worker = CameraFrameProducerWorker(self.newCameraFrameSignal, 10)
        self._camera_frame_producer_thread = QThread()
        self.runWorkerOnNewThread(self._camera_frame_producer_worker, self._camera_frame_producer_thread)

    def startGazeVectorProducer(self):
        self._gaze_vector_producer_worker = GazeVectorProducerWorker(self.newGazeVectorSignal, 10)
        self._gaze_vector_producer_thread = QThread()
        self.runWorkerOnNewThread(self._gaze_vector_producer_worker, self._gaze_vector_producer_thread)

    def runWorkerOnNewThread(self, worker, thread):
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        thread.start()

    def stopCameraFrameProducer(self):
        self._camera_frame_producer_worker.stopSignal.emit()
        self.stopThread(self._camera_frame_producer_thread)
    
    def stopGazeVectorProducer(self):
        self._gaze_vector_producer_worker.stopSignal.emit()
        self.stopThread(self._gaze_vector_producer_thread)
    
    def stopThread(self, thread):
        thread.quit()
        thread.wait()

    def setCameraFrame(self, camera_frame: np.ndarray):
        self._camera_frame = camera_frame
        self.update()
    
    def setGazeVector(self, gaze_vector):
        self._gaze_vector = gaze_vector
        self.update()

    def drawCameraFrame(self, camera_frame: np.ndarray):
        painter = QPainter(self)
        pixmap = self.ndarray_to_qpixmap(camera_frame)
        pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        painter.drawPixmap(71, 0, pixmap)

    def ndarray_to_qpixmap(self, image_array: np.ndarray):
        image_array = image_array.transpose(1, 2, 0)
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        qpixmap = QPixmap.fromImage(qimage)
        return qpixmap

    def drawGazeVector(self, gaze_vector: Tuple):
        eye_positions, gaze = gaze_vector
        image = self._camera_frame
        image = add_gaze_to_image(image, eye_positions, gaze)

        painter = QPainter(self)
        pixmap = self.ndarray_to_qpixmap(image)
        pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        painter.drawPixmap(71, 0, pixmap)
  
    def paintEvent(self, event):
        super().paintEvent(event)
        if self._camera_frame is not None:
            self.drawCameraFrame(self._camera_frame)
        if self._gaze_vector is not None:
            self.drawGazeVector(self._gaze_vector)

    def closeEvent(self, event):
        self.stopCameraFrameProducer()
        self.stopGazeVectorProducer()
        self.windowClosedSignal.emit()
        event.accept()

# class CameraFrameProducerWorker(ProducerWorker):
#     def __init__(self, *args):
#         super().__init__(*args)
#         camera_identifier = "192.168.1.9"
#         self._camera = Camera(f"http://{camera_identifier}:8000/")
#         self.stopSignal.connect(self._camera.close)

#     def _produce(self):
#         camera_frame = self._camera.get_current_frame()
#         return camera_frame

class CameraFrameProducerWorker(ProducerWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self._gazePredictor = GazePredictor()
        self.stopSignal.connect(self._gazePredictor.close)

    def _produce(self):
        camera_frame = self._gazePredictor._camera.get_current_frame()
        return camera_frame


class GazeVectorProducerWorker(ProducerWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self._gazePredictor = GazePredictor()
        self.stopSignal.connect(self._gazePredictor.close)

    def _produce(self):
        prediction = self._gazePredictor.get_gaze_vector_prediction()
        return prediction

