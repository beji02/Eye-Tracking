import numpy as np
import matplotlib.pyplot as plt
import cv2


def add_gaze_to_image(image, eye_positions, gaze, color=(0, 0, 255)):
    image = image.copy()
    image = image.transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # print(image)

    pitch, yaw = gaze
    (x1, y1), (x2, y2) = eye_positions
    # print(x1, y1, x2, y2)

    dx = -1000 * np.sin(pitch) * np.cos(yaw)
    dy = -1000 * np.sin(yaw)
    # print(dx, dy)

    image = draw_arrow(image, x1, y1, dx, dy, color)
    image = draw_arrow(image, x2, y2, dx, dy, color)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)

    return image

def draw_arrow(image, x, y, dx, dy, color=(0, 0, 255)):
    x = int(np.round(x))
    y = int(np.round(y))
    x1 = int(np.round(x + dx))
    y1 = int(np.round(y + dy))
    cv2.arrowedLine(image, (x, y), (x1, y1), color, 2, cv2.LINE_AA, tipLength=0.18)
    return image

