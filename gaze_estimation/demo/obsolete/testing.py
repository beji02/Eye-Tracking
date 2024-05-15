from gui.camera import Camera
import cv2
import numpy as np
from face_detection import RetinaFace
import matplotlib.pyplot as plt
import matplotlib
from demo.visualize import add_gaze_to_image
from demo.models import CombinedModel, GazeEstimationModelWithResnet
from PIL import Image


def print_image(image):
    print("DA")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("DA")
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    print("DA")


def crop_face(image, faces):
    cropped_image = None
    eye_positions = None
    for box, landmarks, score in faces:
        if score > 0.8:
            # print("Box:", box)
            # print("Landmarks:", landmarks)
            # print("Score:", score)

            # Extract bounding box coordinates
            x_min = max(0, int(box[0]))
            y_min = max(0, int(box[1]))
            x_max = int(box[2])
            y_max = int(box[3])

            cropped_image = image[y_min:y_max, x_min:x_max]      
            eye_positions = landmarks[:2]      
    return cropped_image, eye_positions
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# frame = frame.transpose(2, 0, 1)



camera = Camera()
raw_image = None
while raw_image is None:
    raw_image = camera.get_current_frame()
camera.close()
raw_image = raw_image.transpose(1, 2, 0)
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/output/raw_image.jpg', raw_image)
# raw_image = cv2.imread('/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/gloria.jpg')

# raw_image = Image.open('/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/gloria2.jpg')
# raw_image = np.array(raw_image)
# print(raw_image.shape)

# Convert RGB to BGR (OpenCV uses BGR by default)
# raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)


# print_image(raw_image)

face_detector = RetinaFace(gpu_id=0)
faces = face_detector(raw_image)
cropped_image, eye_positions = crop_face(raw_image, faces)
cv2.imwrite('/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/output/cropped_image.jpg', cropped_image)
# print_image(cropped_image)

resized_image = cv2.resize(cropped_image, (448, 448))
cv2.imwrite('/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/output/resized_image.jpg', resized_image)
print_image(resized_image)

print("pe aiic")
resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
resized_image = resized_image.transpose(2, 0, 1)
# image = Image.open("/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/114.jpg")

gaze_estimator = GazeEstimationModelWithResnet()
pitch_predicted, yaw_predicted = gaze_estimator.forward(resized_image)
print(pitch_predicted, yaw_predicted)

raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
raw_image = raw_image.transpose(2, 0, 1)
image = add_gaze_to_image(raw_image, eye_positions, (pitch_predicted, yaw_predicted))
image = image.transpose(1, 2, 0)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite('/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/output/with_gaze.jpg', image)

# print_image(image)








# camera = Camera()

# image = None
# image
# while image is None:
#     image = camera.get_current_frame()
# image = image.transpose(1, 2, 0)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# cv2.imwrite('/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/image_from_array.jpg', image)
# camera.close()



# face_detector = RetinaFace(gpu_id=0)

# # image = cv2.imread('/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/image_from_array.jpg')
# def print_image(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.imshow(image_rgb)
#     plt.axis('off')
#     plt.show()

# image = cv2.imread('/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/face.jpg')
# print_image(image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = image.transpose(2, 0, 1)

# model = CombinedModel()
# eye_positions, gaze = model.forward(image)
# print(image.shape)
# image = add_gaze_to_image(image, eye_positions, gaze)
# image = image.transpose(1, 2, 0)

# plt.imshow(image)
# plt.axis('off')
# plt.show()



# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# image = image.transpose(1, 2, 0)
# faces = face_detector(image)




# def crop_face(image, faces):
#     crops = []
#     for box, landmarks, score in faces:
#         if score > 0.8:
#             print("Box:", box)
#             print("Landmarks:", landmarks)
#             print("Score:", score)

#             # Extract bounding box coordinates
#             x_min = max(0, int(box[0]))
#             y_min = max(0, int(box[1]))
#             x_max = int(box[2])
#             y_max = int(box[3])

#             cropped_image = image[y_min:y_max, x_min:x_max]
#             resized_image = cv2.resize(cropped_image, (224, 224))
#             crops.append(resized_image)
#     return np.array(crops)

# def plot_image_landmarks_bbox(image, faces):
#     for box, landmarks, score in faces:
#         if score > 0.8:
#             print("Box:", box)
#             print("Landmarks:", landmarks)
#             print("Score:", score)

#             # Extract bounding box coordinates
#             x_min = max(0, int(box[0]))
#             y_min = max(0, int(box[1]))
#             x_max = int(box[2])
#             y_max = int(box[3])

#             # Draw bounding box
#             cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#             # Draw landmarks
#             for landmark in landmarks[:2]:
#                 cv2.circle(image, (int(landmark[0]), int(landmark[1])), 2, (255, 0, 0), -1)
    
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     plt.imshow(image_rgb)
#     plt.axis('off')
#     plt.show()

# crops = crop_face(image, faces)
# # print(crops)
# # print(crops.shape)
# print_image(crops[0])
# cv2.imwrite('/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/face.jpg', crops[0])

# plot_image_landmarks_bbox(image, faces)




    


