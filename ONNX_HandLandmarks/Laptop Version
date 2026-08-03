import cv2 as cv
import numpy as np
from math import degrees
import copy
import csv

from model import PalmDetection
from model import HandLandmark
from model import KeyPointClassifier
from utils.utils import rotate_and_crop_rectangle

def pre_process_landmark(landmarks):

    # No landmarks
    if len(landmarks) == 0:
        return []

    # Make a copy so we don't change the original
    landmarks = copy.deepcopy(landmarks)

    # Use the wrist (landmark 0) as the origin
    base_x = landmarks[0][0]
    base_y = landmarks[0][1]

    # Convert every point to coordinates relative to the wrist
    for point in landmarks:
        point[0] = point[0] - base_x
        point[1] = point[1] - base_y

    # Flatten [[x,y],[x,y],...] into [x,y,x,y,...]
    feature_vector = []

    for point in landmarks:
        feature_vector.append(point[0])
        feature_vector.append(point[1])

    # Find the largest absolute value
    max_value = 0

    for value in feature_vector:
        if abs(value) > max_value:
            max_value = abs(value)

    # Normalize everything to [-1, 1]
    if max_value != 0:
        for i in range(len(feature_vector)):
            feature_vector[i] = feature_vector[i] / max_value

    return feature_vector


# Load models
palm_detection = PalmDetection(score_threshold=0.6)
hand_landmark = HandLandmark()
keypoint_classifier = KeyPointClassifier()

with open("model/keypoint_classifier/keypoint_classifier_label.csv") as f:
        labels = [row[0] for row in csv.reader(f)]

cap = cv.VideoCapture(0)
width = 640
height = 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv.flip(frame, 1)
    display = frame.copy()

    # -----------------------------
    # Palm detection
    # -----------------------------
    hands = palm_detection(frame)

    rects = []

    for hand in hands:

        size = hand[0]
        rotation = hand[1]
        cx = hand[2] * width
        cy = hand[3] * height

        rects.append([cx, cy, size * width, size * height, degrees(rotation)])

    pre_processed = []
    if len(rects) > 0:

        rects = np.asarray(rects, dtype=np.float32)

        # -----------------------------
        # Crop + rotate hands
        # -----------------------------
        crops = rotate_and_crop_rectangle(image=frame, rects_tmp=rects, operation_when_cropping_out_of_range='padding')


        # -----------------------------
        # Landmark prediction
        # -----------------------------
        landmarks, sizes = hand_landmark(images=crops, rects=rects)


        # Draw landmarks
        for hand_landmarks in landmarks:

            for x, y in hand_landmarks:

                cv.circle(display, (int(x), int(y)), 5, (255,0,0), -1)

            pre_processed = pre_process_landmark(hand_landmarks)

            #gesture classification

            gesture_id = keypoint_classifier( np.asarray([pre_processed], dtype=np.float32))

            gesture_name = labels[gesture_id[0]]
            cv.putText(display, gesture_name, (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)



    cv.imshow("Hand landmarks", display)

    if cv.waitKey(1) == 27:
        break


cap.release()
cv.destroyAllWindows()

