from config.args import parse_demo_args
from demo.models import CombinedModel
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from data.datasets import InferenceDataset
from data.data_transformation import create_data_transformations_for_resnet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from demo.visualize import add_gaze_to_image
import os
import torch
import time
from gui.camera import Camera
import random
from gaze_utils.gaze_utils import gazeto3d, gazeto3d_demo, gaze3dTo2dCoordinates, gaze3dTo2dCoordinates_custom, gaze3dTo2dCoordinates_with_calib

def get_experiment_path(args):
    experiment_path = args.experiment_path
    return Path(experiment_path)

def get_image_path(args):
    image_path = args.image_path
    return image_path

def get_camera_identifier(args):
    camera_identifier = args.camera_identifier
    if camera_identifier is None:
        pass
    elif len(camera_identifier) == 1:
        camera_identifier = int(camera_identifier)
    else:
        camera_identifier = f"http://{camera_identifier}:8000/"
    return camera_identifier

def run_inference_on_camera_feed(face_detection_model, gaze_estimation_model):
    cv2.imread()

def transform_image_from_cv2_format_to_numpy_rgb_chw(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    return image

def transform_image_from_numpy_rgb_chw_to_cv2_format(image):
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def draw_red_dot(img, x, y):
    cv2.circle(img, (y, x), 5, (0, 0, 255), -1)

def create_white_canvas():
    canvas = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    return canvas

def display_canvas(canvas):
    cv2.namedWindow("Canvas", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Canvas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Canvas", canvas)

def main():
    args = parse_demo_args()
    experiment_path = get_experiment_path(args)
    image_path = get_image_path(args)

    model = CombinedModel(experiment_path)

    if args.demo_type == "preview":
        if image_path is None:
            camera_identifier = get_camera_identifier(args)
            camera = Camera(camera_identifier)
            while True:
                image = camera.get_current_frame()
                if image is not None:
                    eye_positions, gaze = model.forward(image)
                    print(gaze)
                    gaze_3d = gazeto3d(gaze)
                    print("Gaze 3d:", gaze_3d)
                    if gaze is not None:
                        image = add_gaze_to_image(image, eye_positions, gaze)
                    image = transform_image_from_numpy_rgb_chw_to_cv2_format(image)
                    cv2.imshow("Demo", image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            camera.close()
        else:
            image = cv2.imread(image_path)
            image = transform_image_from_cv2_format_to_numpy_rgb_chw(image)
            eye_positions, gaze = model.forward(image)
            print("Gaze yaw/pitch:", gaze)
            gaze_3d = gazeto3d(gaze)
            print("Gaze 3d:", gaze_3d)
            y, x = gaze3dTo2dCoordinates(gaze_3d)
            print("Gaze 2d:", y, x)
            image = add_gaze_to_image(image, eye_positions, gaze)
            image = transform_image_from_numpy_rgb_chw_to_cv2_format(image)
            cv2.imwrite(str(Path(os.getcwd()) / "demo" / "output" / "image_with_gaze.jpg"), image)
    elif args.demo_type == "canvas":
        if image_path is None:
            camera_identifier = get_camera_identifier(args)
            camera = Camera(camera_identifier)
            while True:
                image = camera.get_current_frame()
                if image is not None:
                    canvas = create_white_canvas()
                    _, gaze = model.forward(image)
                    if gaze is not None:
                        
                        gaze_3d = gazeto3d(gaze)
                        print("Gaze 3d:", gaze_3d)
                        y, x = gaze3dTo2dCoordinates_custom(gaze_3d)
                        
                        print(x, y)
                        # x, y = get_screen_coordinates_from_gaze(gaze)
                        # x = random.randint(0, canvas.shape[1])
                        # y = random.randint(0, canvas.shape[0])
                        draw_red_dot(canvas, int(x), int(y))
                    
                    display_canvas(canvas)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
            camera.close()












    

    
    
    

# canvas = np.ones((1080, 1920, 3), dtype=np.uint8) * 255


# draw_red_dot(canvas, 100, 100)
# draw_red_dot(canvas, 0, 0)
# draw_red_dot(canvas, 1080, 1920)

# # Display fullscreen
# cv2.namedWindow("Fullscreen", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Fullscreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.imshow("Fullscreen", canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ == "__main__":
    main()