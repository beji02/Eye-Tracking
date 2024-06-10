from PyQt5.QtWidgets import (
    QMainWindow,
    QAction,
    QApplication,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
)
from PyQt5 import QtGui
import sys
from PyQt5.QtCore import pyqtSlot, Qt, pyqtSignal
import sys
import random
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import QTimer, QThread, QObject
import threading
from abc import ABC, abstractmethod
from gui.gaze_predictor import GazePredictor
from gui.gui_windows.ProducerWorker import ProducerWorker
import time
from PyQt5.QtGui import QFont, QColor

class TracingWindow(QMainWindow):
    newTargetSignal = pyqtSignal(tuple)
    newGazeSignal = pyqtSignal(tuple)
    newFpsSignal = pyqtSignal(float)
    windowClosedSignal = pyqtSignal()

    def __init__(self, camera_identifier, experiment_path):
        super().__init__()
        self._camera_identifier = camera_identifier
        self._experiment_path = experiment_path
        self._gaze = None
        self._target = None
        self._fps = None
        self._object_radius = 10
        
        self.newTargetSignal.connect(self.setTarget)
        self.newGazeSignal.connect(self.setGaze)
        self.newFpsSignal.connect(self.setFps)

        self.fps_label = QLabel("", self)
        self.fps_label.setFont(QFont('Arial', 40))
        self.fps_label.setStyleSheet('color: green')
        self.fps_label.move(100, 10)

        self.startTargetProducer()
        self.startGazeProducer()
        self.startFpsProducer()

        self.setWindowTitle("Tracing")
        self.showMaximized()

    def startFpsProducer(self):
        self._fps_producer_worker = FpsProducerWorker(self._camera_identifier, self._experiment_path, self.newFpsSignal, 1)
        self._fps_producer_thread = QThread()
        self.runWorkerOnNewThread(self._fps_producer_worker, self._fps_producer_thread)

    def startTargetProducer(self):
        self._target_producer_worker = TargetProducerWorker(self.newTargetSignal, 3000)
        self._target_producer_thread = QThread()
        self.runWorkerOnNewThread(self._target_producer_worker, self._target_producer_thread)

    def startGazeProducer(self):
        self._gaze_producer_worker = GazeProducerWorker(self._camera_identifier, self._experiment_path, self.newGazeSignal, 1)
        self._gaze_producer_thread = QThread()
        self.runWorkerOnNewThread(self._gaze_producer_worker, self._gaze_producer_thread)
    
    def runWorkerOnNewThread(self, worker, thread):
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        thread.start()

    def stopTargerProducer(self):
        self._target_producer_worker.stopSignal.emit()
        self.stopThread(self._target_producer_thread)
    
    def stopGazeProducer(self):
        self._gaze_producer_worker.stopSignal.emit()
        self.stopThread(self._gaze_producer_thread)
    
    def stopFpsProducer(self):
        self._fps_producer_worker.stopSignal.emit()
        self.stopThread(self._fps_producer_thread)
    
    def stopThread(self, thread):
        thread.quit()
        thread.wait()

    def setGaze(self, gaze):
        self._gaze = gaze
        self.update()

    def setTarget(self, target):
        self._target = target
        self.update()
    
    def setFps(self, fps):
        self._fps = fps

    def drawPoint(self, color, position):
        painter = QPainter(self)
        painter.setBrush(color)
        painter.drawEllipse(
            position[0] - self._object_radius,
            position[1] - self._object_radius,
            2 * self._object_radius,
            2 * self._object_radius,
        )
    
    def drawFps(self, fps: float):
        self.fps_label.setText(f'Inference avg time: {fps:.5f}')
        self.fps_label.adjustSize()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._gaze is not None:
            self.drawPoint(QColor(255, 0, 0), self._gaze)
        if self._target is not None:
            self.drawPoint(QColor(0, 0, 255), self._target)
        if self._fps is not None:
            self.drawFps(self._fps)
    
    def closeEvent(self, event):
        self.stopTargerProducer()
        self.stopGazeProducer()
        self.stopFpsProducer()
        self.windowClosedSignal.emit()
        event.accept()

class TargetProducerWorker(ProducerWorker):
    def _produce(self):
        position = (random.randint(50, 1870), random.randint(50, 950))
        return position
    
class GazeProducerWorker(ProducerWorker):
    def __init__(self, camera_identifier, experiment_path, *args):
        super().__init__(*args)
        self._gazePredictor = GazePredictor(camera_identifier, experiment_path)
        self.stopSignal.connect(self._gazePredictor.close)
    
    def _produce(self):
        gaze = self._gazePredictor.get_on_screen_prediction()
        return gaze
    
class FpsProducerWorker(ProducerWorker):
    def __init__(self, camera_identifier, experiment_path, *args):
        super().__init__(*args)
        self._gazePredictor = GazePredictor(camera_identifier, experiment_path)
        self.stopSignal.connect(self._gazePredictor.close)

    def _produce(self):
        fps = self._gazePredictor.get_gaze_estimator_fps()
        return fps