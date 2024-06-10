import numpy as np
from gaze_utils.gazeconversion import Gaze3DTo2D
import scipy.io as sio
import sys
import cv2


def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


def gazeto3d_demo(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = np.sin(gaze[1])
    gaze_gt[2] = np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


def angular(gaze, label):
    total = np.sum(gaze * label)
    return (
        np.arccos(
            min(total / (np.linalg.norm(gaze) * np.linalg.norm(label)), 0.9999999)
        )
        * 180
        / np.pi
    )


def gaze3dTo2dCoordinates_with_calib(gaze):
    upper_left_gaze = np.array([-0.194969, 0.10294703, -0.97539169])
    lower_right_gaze = np.array([-0.04411286, 0.18820287, -0.981139])

    w_pixel = 1920
    h_pixel = 1080

    percent_left = (gaze[0] - upper_left_gaze[0]) / (lower_right_gaze[0] - upper_left_gaze[0])
    x_gaze = w_pixel * percent_left
    percent_up = (gaze[1] - upper_left_gaze[1]) / (lower_right_gaze[1] - upper_left_gaze[1])
    y_gaze = h_pixel * percent_up

    return x_gaze, y_gaze


def gaze3dTo2dCoordinates_custom(gaze):
    print(gaze)
    origin = np.array(
        [0, 0, 500]
    )  # 500: distanta fata-camera, first 0: distanta spre stg/dr, second 0: distanta sus jos
    tvec = np.array([175, 15, 0])
    rmat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])

    w_pixel = 1920
    h_pixel = 1080
    w_mm = 342
    h_mm = 194
    w_ratio = w_pixel / w_mm
    h_ratio = h_pixel / h_mm

    gaze[1] = -gaze[1]
    const_for_gaze_line_with_screen_plane_eq = -origin[2] / gaze[2]
    # print(const_for_gaze_line_with_screen_plane_eq)
    x_mm_intersection_gaze_line_with_screen_plane = (
        const_for_gaze_line_with_screen_plane_eq * gaze[0]
    )
    # print(x_mm_intersection_gaze_line_with_screen_plane)
    y_mm_intersection_gaze_line_with_screen_plane = (
        const_for_gaze_line_with_screen_plane_eq * gaze[1]
    )
    
    print("Xmm:", x_mm_intersection_gaze_line_with_screen_plane)
    print("Ymm:", y_mm_intersection_gaze_line_with_screen_plane)   

    # print(y_mm_intersection_gaze_line_with_screen_plane)


    x_mm = x_mm_intersection_gaze_line_with_screen_plane*2 + tvec[0]
    # print(x_mm)
    y_mm = y_mm_intersection_gaze_line_with_screen_plane*2 + tvec[1]
    # print(y_mm)
    x_mm = rmat[0][0] * x_mm
    # print(x_mm)
    y_mm = rmat[1][1] * y_mm
    # print(y_mm)

    x_pixel = x_mm * w_ratio
    y_pixel = y_mm * h_ratio

    # x_pixel = x_pixel * (1 + 2*abs(x_pixel - w_pixel/2)/w_pixel)
    # y_pixel = y_pixel + (1 + 2*abs(y_pixel - h_pixel/2)/w_pixel)

    print("Xpixel:", x_pixel)
    print("Ypixel:", y_pixel)

    return (x_pixel, y_pixel)


def gaze3dTo2dCoordinates(gaze):
    # gaze_yaw_pitch = np.array([0.0679, -0.1258])
    # gaze = gazeto3d(gaze_yaw_pitch)
    origin = np.array(
        [0, 0, 500]
    )  # 500: distanta fata-camera, first 0: distanta spre stg/dr, second 0: distanta sus jos

    monitorpose = sio.loadmat(
        "/mnt/d/Downloads/MPIIFaceGaze/MPIIFaceGaze/p10/Calibration/monitorPose.mat"
    )
    # print(monitorpose)
    rvec = monitorpose["rvects"]
    # print(rvec)
    tvec = monitorpose["tvecs"]  # [175 de la st la dr, 15 vertical, 0 spre mine ]
    # print(tvec.shape)
    tvec = np.array([-175, -15, 0])
    # print(tvec.shape)
    rmat = cv2.Rodrigues(rvec)[0]
    # print(rmat.shape)
    rmat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    # print(rmat.shape)

    screen = sio.loadmat(
        "/mnt/d/Downloads/MPIIFaceGaze/MPIIFaceGaze/p00/Calibration/screenSize.mat"
    )
    w_pixel = screen["width_pixel"][0][0]  # 1080
    h_pixel = screen["height_pixel"][0][0]  # 1920
    w_pixel = 1080
    h_pixel = 1920
    # print(w_pixel, h_pixel)
    w_mm = screen["width_mm"][0][0]  # pt mine e 342 mm
    h_mm = screen["height_mm"][0][0]  # 194 mm
    w_mm = 342
    h_mm = 194
    # print(w_mm, h_mm)
    w_ratio = w_pixel / w_mm
    h_ratio = h_pixel / h_mm

    x_mm, y_mm = Gaze3DTo2D(gaze, origin, rmat, tvec)
    x_pixel = x_mm * w_ratio
    y_pixel = y_mm * h_ratio
    # print(x_mm, y_mm)
    # print(x_pixel, y_pixel)
    return (x_pixel, y_pixel)


if __name__ == "__main__":
    gaze3dTo2dCoordinates(None)
