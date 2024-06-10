from abc import ABC, abstractmethod
import numpy as np
import time

class ITimer(ABC):
    @abstractmethod
    def get_fps(self):
        pass
    
class AbsTimer(ITimer):
    def __init__(self):
        self._no_events = 0
        self._total_time = 0

    def get_fps(self):
        if self._no_events == 0:
            return 0
        return self._total_time / self._no_events

    def start_timer(self):
        self._start_time = time.time()
    
    def end_timer(self):
        end_time = time.time()
        time_dif = end_time - self._start_time
        self._no_events += 1
        self._total_time += time_dif


class IModel(ABC):
    @abstractmethod
    def forward(self, image: np.ndarray):
        pass


class ModelWithTimer(AbsTimer, IModel):
    def __init__(self, model: IModel):
        AbsTimer.__init__(self)
        self._model = model

    def forward(self, image: np.ndarray):
        self.start_timer()
        output = self._model.forward(image)
        self.end_timer()
        return output

    
        