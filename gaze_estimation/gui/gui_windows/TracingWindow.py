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
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import QTimer, QThread, QObject
import threading
from abc import ABC, abstractmethod
from gui.gaze_predictor import GazePredictor
from gui.gui_windows.ProducerWorker import ProducerWorker

class TracingWindow(QMainWindow):
    newTargetSignal = pyqtSignal(tuple)
    newGazeSignal = pyqtSignal(tuple)
    windowClosedSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._gaze = None
        self._target = None
        self._object_radius = 10
        
        self.newTargetSignal.connect(self.setTarget)
        self.newGazeSignal.connect(self.setGaze)

        self.startTargetProducer()
        # self.startGazeProducer()

        self.setWindowTitle("Tracing")
        self.showMaximized()

    def startTargetProducer(self):
        self._target_producer_worker = TargetProducerWorker(self.newTargetSignal, 1000)
        self._target_producer_thread = QThread()
        self.runWorkerOnNewThread(self._target_producer_worker, self._target_producer_thread)

    def startGazeProducer(self):
        self._gaze_producer_worker = GazeProducerWorker(self.newGazeSignal, 1000)
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
    
    def stopThread(self, thread):
        thread.quit()
        thread.wait()

    def setGaze(self, gaze):
        self._gaze = gaze
        self.update()

    def setTarget(self, target):
        self._target = target
        self.update()

    def drawPoint(self, color, position):
        painter = QPainter(self)
        painter.setBrush(color)
        painter.drawEllipse(
            position[0] - self._object_radius,
            position[1] - self._object_radius,
            2 * self._object_radius,
            2 * self._object_radius,
        )

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._gaze is not None:
            self.drawPoint(QColor(255, 0, 0), self._gaze)
        if self._target is not None:
            self.drawPoint(QColor(0, 0, 255), self._target)
    
    def closeEvent(self, event):
        self.stopTargerProducer()
        self.stopGazeProducer()
        self.windowClosedSignal.emit()
        event.accept()

class TargetProducerWorker(ProducerWorker):
    def _produce(self):
        position = (random.randint(50, 950), random.randint(50, 600))
        return position
    
class GazeProducerWorker(ProducerWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self._gazePredictor = GazePredictor()
        self.stopSignal.connect(self._gazePredictor.close)
    
    def _produce(self):
        gaze = self._gazePredictor.get_on_screen_prediction()
        return gaze
