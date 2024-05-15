import argparse


def parse_experiment_args():
    parser = argparse.ArgumentParser(description="Train script for gaze estimation.")
    parser.add_argument(
        "-experiment_path",
        dest="experiment_path",
        help="Path to the experiment to run. This folder should contain a config.json file. After training, an output directory will be created in this directory.",
        type=str,
        required=True
    )
    args = parser.parse_args()
    return args

def parse_demo_args():
    parser = argparse.ArgumentParser(description="Demo script for gaze estimation.")
    parser.add_argument(
        "-demo_type",
        dest="demo_type",
        help="What type of demo do you want: preview/canvas.",
        type=str,
        default="preview"
    )
    parser.add_argument(
        "-experiment_path",
        dest="experiment_path",
        help="Path to the experiment from where the model to be run.",
        type=str,
        required=True
    )
    parser.add_argument(
        "-image_path",
        dest="image_path",
        help="Path to the image to be run on.",
        type=str,
        default=None
    )
    parser.add_argument(
        "-camera_identifier",
        dest="camera_identifier",
        help="IP or integer representing camera identifier.",
        type=str,
        default=None
    )
    args = parser.parse_args()
    return args
