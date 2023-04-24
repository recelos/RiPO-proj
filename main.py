import cv2 as cv 
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

cv.startWindowThread()

vid = cv.VideoCapture("recordings/video.mp4")

while True:
        _, inp = vid.read()
        inp = cv.resize(inp, (480,640))

        tensor = tf.convert_to_tensor(inp, dtype=tf.uint8)
        tensor = tf.expand_dims(tensor , 0)

        boxes, scores, classes, num_detections = detector(tensor)

        pred_labels = classes.numpy().astype('int')[0] 
        pred_labels = [labels[i] for i in pred_labels]
        pred_boxes = boxes.numpy()[0].astype('int')
        pred_scores = scores.numpy()[0]

        for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
            if score < 0.5:
                continue

            score_txt = f'{100 * round(score)}%'
            img_boxes = cv.rectangle(inp, (xmin, ymax),(xmax, ymin),(0,255,0),2)      
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img_boxes, label,(xmin, ymax-10), font, 1.5, (255,0,0), 2, cv.LINE_AA)
            cv.putText(img_boxes,score_txt,(xmax, ymax-10), font, 1.5, (255,0,0), 2, cv.LINE_AA)
        cv.imshow('frame', inp)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv.destroyAllWindows()
cv.waitKey(1)
        