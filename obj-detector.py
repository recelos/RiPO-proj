import cv2 as cv 
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import mediapipe as mp
import numpy as np

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.1)

cv.startWindowThread()

vid = cv.VideoCapture("video.mp4")

frameskip = 5
frame = 0
while True:
        if frame < frameskip:
             frame += 1
             continue
        else:
             frame = 0
        _, inp = vid.read()

        tensor = tf.convert_to_tensor(inp, dtype=tf.uint8)
        tensor = tf.expand_dims(tensor , 0)

        boxes, scores, classes, num_detections = detector(tensor)

        pred_labels = classes.numpy().astype('int')[0] 
        pred_labels = [labels[i] for i in pred_labels]
        pred_boxes = boxes.numpy()[0].astype('int')
        pred_scores = scores.numpy()[0]

        # convert the frame to RGB format
        RGB = cv.cvtColor(inp, cv.COLOR_BGR2RGB)

        # process the RGB frame to get the result
        results = pose.process(RGB)

        # draw detected skeleton on the frame
        mp_drawing.draw_landmarks(
            inp, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
            if score < 0.45:
                continue

            score_txt = f'{100 * round(score)}%'
            img_boxes = cv.rectangle(inp, (xmin, ymax),(xmax, ymin),(0,255,0),2)      
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img_boxes, label,(xmin, ymax-10), font, 1.5, (255,0,0), 2, cv.LINE_AA)
            cv.putText(img_boxes,score_txt,(xmax, ymax-10), font, 1.5, (255,0,0), 2, cv.LINE_AA)
        inp = cv.resize(inp, (480,640))
        cv.imshow('frame', inp)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv.destroyAllWindows()
cv.waitKey(1)