from gui.camera import Camera
from demo.models import CombinedModel as Model
import random
import threading
from gui.Singleton import SingletonABC
from pathlib import Path
import time
from gaze_utils.gaze_utils import gazeto3d, gaze3dTo2dCoordinates_custom

'''
predicts yaw and pitch of eye gaze based on an image of a person
input:
image: np.ndarray - RGB, CxHxW representation of an image
output:
np.ndarray - an array with two float values representing yaw and gaze
'''
class GazePredictor(SingletonABC):
    def _initialize(self, camera_identifier, experiment_path):
        self._camera = Camera(f"http://{camera_identifier}:8000/")
        experiment_path = Path(experiment_path)
        self._model = Model(experiment_path)
        self._lock = threading.Lock()

    def get_gaze_estimator_fps(self):
        self._lock.acquire()
        fps = self._model.get_fps()
        self._lock.release()
        return fps

    def get_on_screen_prediction(self):
        self._lock.acquire()
        image = self._camera.get_current_frame()
        gaze = None
        if image is not None:
            prediction = self._model.forward(image)
            if prediction is not None:
                _, gaze = prediction
                gaze_3d = gazeto3d(gaze)
                # print("Gaze 3d:", gaze_3d)
                y, x = gaze3dTo2dCoordinates_custom(gaze_3d)
                # print(y, x)
                y, x = int(y), int(x)
                gaze = (y, x)
        self._lock.release()
        return gaze
    
    def get_gaze_vector_prediction(self):
        self._lock.acquire()
        image = self._camera.get_current_frame()
        prediction = None
        if image is not None:
            prediction = self._model.forward(image)
        # print(prediction)
        # prediction = self.process_prediction_for_screen_display(prediction)
        self._lock.release()
        return prediction
    
    def get_camera_current_frame(self):
        self._lock.acquire()
        image = self._camera.get_current_frame()
        self._lock.release()
        return image
    
    def process_prediction_for_screen_display(self, prediction):
        return (random.randint(50, 950), random.randint(50, 600))
    
    def _destruct(self) -> None:
        self._camera.close()

    @classmethod
    def _reset_instance(cls):
        cls._instance = None

# gaze_pred = GazePredictor()
# import time
# # time.sleep(2)
# # gaze_pred.close()

# gaze_pred = GazePredictor()
# gaze_pred2 = GazePredictor()
# gaze_pred.close()
# time.sleep(5)
# gaze_pred2.close()



