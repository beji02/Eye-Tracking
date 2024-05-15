from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
import cv2
import threading
import time
import matplotlib.pyplot as plt

class SingletonABC(ABC):
    _instance = None
    _reference_count = 0
    _singleton_lock = None
    _initialized = False

    def __init__(self, *args, **kwargs):
        if not self._initialized:
            self._initialize(*args, **kwargs)
            self._initialized = True

    @classmethod
    def __new__(cls, *args, **kwargs):
        if cls._singleton_lock is None:
            cls._singleton_lock = threading.Lock()
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            cls._reference_count += 1
            return cls._instance
    
    def close(self) -> None:
        cls = self.__class__
        with cls._singleton_lock:
            cls._reference_count -= 1
            if cls._reference_count == 0:
                self._destruct()
                cls._instance = None
                cls._initialized = False
    
    @abstractmethod
    def _destruct(self) -> None:
        pass

    @abstractmethod
    def _initialize(self) -> None:
        pass

# class A(SingletonABC):
#     def _destruct(self) -> None:
#         pass

# class C(SingletonABC):
#     def _initialize(self) -> None:
#         return super()._initialize()
#     def _destruct(self) -> None:
#         pass

# class B(SingletonABC):
#     def _initialize(self):
#         self._a = A()

#     def _destruct(self) -> None:
#         pass
#         self._a.close()

# b = B()
# b.close()

# a = A()
# a2 = A()
# c = C()
# c2 = C()
# c3 = C()
# print(A._singleton_lock is C._singleton_lock)
# print(A._reference_count is C._reference_count)
# print(A._reference_count)
# print(C._reference_count)
# print(A._instance is C._instance)

# import time
# time.sleep(1)
# a.close()
# time.sleep(1)
# a.close()

