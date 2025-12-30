import numpy as np
import cv2
from preprocessing import execute
from preprocessing import homography_matrix

def video(path):
    cap = cv2.VideoCapture(path)
    # _, frame = cap.read()
    # M, Minv = homography_matrix(frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))
        M, Minv = homography_matrix(frame)
        result = execute(frame, M, Minv)

        cv2.imshow("lane detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    video("../data/lane_video.mp4")