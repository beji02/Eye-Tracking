import threading
from PyQt5.QtCore import QTimer
from abc import ABC, abstractmethod
from collections.abc import Iterable
from PyQt5.QtCore import QThread, pyqtSignal, QObject

class ProducerWorker(QObject):
    stopSignal = pyqtSignal()

    def __init__(self, signal, time):
        super().__init__()
        self._signal = signal
        self._time = time

    def run(self):
        self._timer = QTimer()
        self.stopSignal.connect(self._timer.stop)
        self._timer.timeout.connect(self._produce_and_send)
        self._timer.start(self._time)
        self._produce_and_send()
    
    def _produce_and_send(self):
        product = self._produce()
        if product is not None:
            self._signal.emit(product)
        
    @abstractmethod
    def _produce(self):
        pass

    