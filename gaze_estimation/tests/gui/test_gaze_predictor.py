#python -m unittest

from gui.gaze_predictor import GazePredictor
import unittest
from unittest.mock import patch
import cv2
from demo.main import transform_image_from_cv2_format_to_numpy_rgb_chw

class TestGazePredictor(unittest.TestCase):

    def setUp(self):
        GazePredictor._reset_instance()
    
    @patch('gui.gaze_predictor.Camera')
    @patch('gui.gaze_predictor.Model')
    def test_single_instantiation(self, MockModel, MockCamera):
        g1 = GazePredictor(camera_identifier='test_camera', experiment_path='/test/path')
        g2 = GazePredictor(camera_identifier='test_camera', experiment_path='/test/path')
        
        assert g1 is g2

    # testeaza get fps (model fps should be called at least once)
    @patch('gui.gaze_predictor.Camera')
    @patch('gui.gaze_predictor.Model')
    def test_get_gaze_estimator_fps(self, MockModel, MockCamera):
        mock_model_instance = MockModel.return_value
        
        g = GazePredictor(camera_identifier='test_camera', experiment_path='/test/path')
        _ = g.get_gaze_estimator_fps()
        mock_model_instance.get_fps.assert_called_once()

    # testeaza get camera current frame (camera -get current frame should be called at least once)
    @patch('gui.gaze_predictor.Camera')
    @patch('gui.gaze_predictor.Model')
    def test_get_camera_current_frame(self, MockModel, MockCamera):
        mock_camera_instance = MockCamera.return_value
        
        g = GazePredictor(camera_identifier='test_camera', experiment_path='/test/path')
        _ = g.get_camera_current_frame()
        mock_camera_instance.get_current_frame.assert_called_once()

    # Integration testing 
    # testeaza cu experiment_path gresit -> eroare
    @patch('gui.gaze_predictor.Camera')
    def test_invalid_experiment_path(self, MockCamera):
        
        with self.assertRaises(FileNotFoundError):
            GazePredictor(camera_identifier='test_camera', experiment_path='/invalid/path')

    # testeaza cu experiment_path corect -> none
    @patch('gui.gaze_predictor.Camera')
    def test_valid_experiment_path_with_dark_image(self, MockCamera):
        mock_camera_instance = MockCamera.return_value

        # load image
        image = cv2.imread("/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/tests/test_data/dark.jpg")
        image = transform_image_from_cv2_format_to_numpy_rgb_chw(image)
        mock_camera_instance.get_current_frame.return_value = image
        
        g = GazePredictor(camera_identifier='test_camera', experiment_path='/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/experiments/Mobilent_lr_00001_strat_all')
        result = g.get_gaze_vector_prediction()
        self.assertIsNone(result)

    # testeaza cu experiment_path corect dar fara fata in imagine -> result
    @patch('gui.gaze_predictor.Camera')
    def test_valid_experiment_path_with_face_image(self, MockCamera):
        mock_camera_instance = MockCamera.return_value

        # load image
        image = cv2.imread("/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/tests/test_data/face.jpg")
        image = transform_image_from_cv2_format_to_numpy_rgb_chw(image)
        mock_camera_instance.get_current_frame.return_value = image
        
        g = GazePredictor(camera_identifier='test_camera', experiment_path='/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/experiments/Mobilent_lr_00001_strat_all')
        result = g.get_gaze_vector_prediction()
        self.assertIsNotNone(result)

    # white box
    # verify lock is aquired before and forgot after calls
    @patch('gui.gaze_predictor.Camera')
    @patch('gui.gaze_predictor.Model')
    @patch('threading.Lock')
    def test_lock_usage(self, MockLock, MockModel, MockCamera):
        mock_lock_instance = MockLock.return_value
        
        g = GazePredictor(camera_identifier='test_camera', experiment_path='/test/path')
        _ = g.get_gaze_estimator_fps()
        self.assertEqual(mock_lock_instance.acquire.call_count, 1)
        self.assertEqual(mock_lock_instance.release.call_count, 1)

        _ = g.get_gaze_vector_prediction()
        self.assertEqual(mock_lock_instance.acquire.call_count, 2)
        self.assertEqual(mock_lock_instance.release.call_count, 2)

        _ = g.get_camera_current_frame()
        self.assertEqual(mock_lock_instance.acquire.call_count, 3)
        self.assertEqual(mock_lock_instance.release.call_count, 3)