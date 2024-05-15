from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
import cv2
import threading
import time
import matplotlib.pyplot as plt
from gui.Singleton import SingletonABC

class ICamera(ABC):
    @abstractmethod
    def get_current_frame(self):
        pass


class MockCamera(ICamera):
    def get_current_frame(self):
        random_image = np.random.rand(720, 1280, 3)
        return random_image


class Camera(ICamera, SingletonABC):
    def _initialize(self, camera_identifier):
        self._camera_identifier = camera_identifier
        self._current_frame = None

        self._current_frame_lock = threading.Lock()
        self._camera_thread = threading.Thread(target=self._record_frames)
        self._event_close = threading.Event()
        self._camera_thread.start()

    def _record_frames(self) -> None:
        cap = cv2.VideoCapture()
        cap.open(self._camera_identifier)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        while not self._event_close.is_set():
            success, frame = cap.read()
            if success:
                frame_array = np.asarray(frame)
                self._update_current_frame(frame_array)
            else:
                cap.release()
                cap.open(self._camera_identifier)
        cap.release()
    
    def _update_current_frame(self, new_frame: np.ndarray) -> None:
        self._current_frame_lock.acquire()
        self._current_frame = new_frame
        self._current_frame_lock.release()
    
    def get_current_frame(self) -> np.ndarray:
        self._current_frame_lock.acquire()
        if self._current_frame is not None:
            frame = self._current_frame.copy()
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.transpose(2, 0, 1)
        else:
            frame = None
        self._current_frame_lock.release()
        return frame
    
    def _destruct(self) -> None:
        self._event_close.set()
        self._camera_thread.join()

# camera = Camera()
# camera2 = Camera()

# camera.close()
# time.sleep(2)
# camera2.close()

# camera = Camera()
# camera2 = Camera()

# camera.close()
# camera2.close()