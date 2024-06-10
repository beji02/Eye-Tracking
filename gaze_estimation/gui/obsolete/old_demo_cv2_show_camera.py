# from MockPredictor import MockPredictor
# import numpy as np
# import cv2

# gaze_predictor = MockPredictor()
# print(gaze_predictor.predict(np.ndarray([])))





# cap = cv2.VideoCapture()
# cap.open("http://192.168.100.58:8000/")

# if not cap.isOpened():
#     raise IOError("Cannot open webcam")

# while True:
#     success, frame = cap.read()

#     frame_array = np.asarray(frame)
#     print(frame_array.shape)

#     cv2.imshow("Demo", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
