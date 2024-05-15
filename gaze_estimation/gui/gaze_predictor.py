from gui.camera import Camera
from demo.models import CombinedModel as Model
import random
import threading
from gui.Singleton import SingletonABC
from pathlib import Path

'''
predicts yaw and pitch of eye gaze based on an image of a person
input:
image: np.ndarray - RGB, CxHxW representation of an image
output:
np.ndarray - an array with two float values representing yaw and gaze
'''
class GazePredictor(SingletonABC):
    def _initialize(self):
        camera_identifier = "192.168.1.9"
        self._camera = Camera(f"http://{camera_identifier}:8000/")
        experiment_path = Path("/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/experiments/test_poti_sa_te_joci")
        self._model = Model(experiment_path)
        self._lock = threading.Lock()

    def get_on_screen_prediction(self):
        self._lock.acquire()
        image = self._camera.get_current_frame()
        prediction = self._model.forward(image)
        prediction = self.process_prediction_for_screen_display(prediction)
        # print(prediction)
        self._lock.release()
        return prediction
    
    def get_gaze_vector_prediction(self):
        self._lock.acquire()
        image = self._camera.get_current_frame()
        if image is not None:
            prediction = self._model.forward(image)
        else:
            prediction = None
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

# gaze_pred = GazePredictor()
# import time
# # time.sleep(2)
# # gaze_pred.close()

# gaze_pred = GazePredictor()
# gaze_pred2 = GazePredictor()
# gaze_pred.close()
# time.sleep(5)
# gaze_pred2.close()



