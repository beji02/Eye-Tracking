from PyQt5.QtWidgets import (
    QMainWindow,
    QAction,
    QApplication,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel
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
from PyQt5.QtGui import QFont, QColor

class WebcamWindow(QMainWindow):
    newCameraFrameSignal = pyqtSignal(np.ndarray)
    newGazeVectorSignal = pyqtSignal(tuple)
    newFpsSignal = pyqtSignal(float)
    windowClosedSignal = pyqtSignal()

    def __init__(self, camera_identifier, experiment_path):
        super().__init__()
        self._camera_identifier = camera_identifier
        self._experiment_path = experiment_path
        self._camera_frame = None
        self._gaze_vector = None
        self._fps = None
        self.newCameraFrameSignal.connect(self.setCameraFrame)
        self.newGazeVectorSignal.connect(self.setGazeVector)
        self.newFpsSignal.connect(self.setFps)

        self.fps_label = QLabel("", self)
        self.fps_label.setFont(QFont('Arial', 40))
        self.fps_label.setStyleSheet('color: green')
        self.fps_label.move(100, 10)

        self.startCameraFrameProducer()
        self.startGazeVectorProducer()
        self.startFpsProducer()

        self.setWindowTitle("Webcam")
        self.showMaximized()

    def startFpsProducer(self):
        self._fps_producer_worker = FpsProducerWorker(self._camera_identifier, self._experiment_path, self.newFpsSignal, 1)
        self._fps_producer_thread = QThread()
        self.runWorkerOnNewThread(self._fps_producer_worker, self._fps_producer_thread)

    def startCameraFrameProducer(self):
        self._camera_frame_producer_worker = CameraFrameProducerWorker(self._camera_identifier, self._experiment_path, self.newCameraFrameSignal, 1)
        self._camera_frame_producer_thread = QThread()
        self.runWorkerOnNewThread(self._camera_frame_producer_worker, self._camera_frame_producer_thread)

    def startGazeVectorProducer(self):
        self._gaze_vector_producer_worker = GazeVectorProducerWorker(self._camera_identifier, self._experiment_path, self.newGazeVectorSignal, 1)
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

    def stopFpsProducer(self):
        self._fps_producer_worker.stopSignal.emit()
        self.stopThread(self._fps_producer_thread)
    
    def stopThread(self, thread):
        thread.quit()
        thread.wait()

    def setCameraFrame(self, camera_frame: np.ndarray):
        if camera_frame is not None:
            self._camera_frame = camera_frame
            self.update()
        
    def setGazeVector(self, gaze_vector):
        if gaze_vector is not None:
            self._gaze_vector = gaze_vector
            self.update()

    def setFps(self, fps):
        self._fps = fps

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

    def drawFps(self, fps: float):
        self.fps_label.setText(f'Inference avg time: {fps:.5f}')
        self.fps_label.adjustSize()
  
    def paintEvent(self, event):
        super().paintEvent(event)
        if self._camera_frame is not None:
            self.drawCameraFrame(self._camera_frame)
        if self._gaze_vector is not None:
            print(self._gaze_vector)
            self.drawGazeVector(self._gaze_vector)
        if self._fps is not None:
            self.drawFps(self._fps)

    def closeEvent(self, event):
        self.stopCameraFrameProducer()
        self.stopGazeVectorProducer()
        self.stopFpsProducer()
        self.windowClosedSignal.emit()
        event.accept()

class CameraFrameProducerWorker(ProducerWorker):
    def __init__(self, camera_identifier, experiment_path, *args):
        super().__init__(*args)
        self._gazePredictor = GazePredictor(camera_identifier, experiment_path)
        self.stopSignal.connect(self._gazePredictor.close)

    def _produce(self):
        camera_frame = self._gazePredictor._camera.get_current_frame()
        return camera_frame


class GazeVectorProducerWorker(ProducerWorker):
    def __init__(self, camera_identifier, experiment_path, *args):
        super().__init__(*args)
        self._gazePredictor = GazePredictor(camera_identifier, experiment_path)
        self.stopSignal.connect(self._gazePredictor.close)

    def _produce(self):
        prediction = self._gazePredictor.get_gaze_vector_prediction()
        return prediction


class FpsProducerWorker(ProducerWorker):
    def __init__(self, camera_identifier, experiment_path, *args):
        super().__init__(*args)
        self._gazePredictor = GazePredictor(camera_identifier, experiment_path)
        self.stopSignal.connect(self._gazePredictor.close)

    def _produce(self):
        fps = self._gazePredictor.get_gaze_estimator_fps()
        return fps
