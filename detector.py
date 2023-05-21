from roboflow import Roboflow
import cv2
import base64
import json
import requests
import time
import asyncio
import numpy as np
import httpx


# load config
with open('./roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    ROBOFLOW_MODEL_ID = config["ROBOFLOW_MODEL_ID"]
    ROBOFLOW_VERSION_NUMBER = config["ROBOFLOW_VERSION_NUMBER"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

# Construct the Roboflow Infer URL
# (if running locally replace https://classify.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL_ID, "/",
    ROBOFLOW_VERSION_NUMBER,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=json",
    "&stroke=5"
    ])

# Get webcam interface via opencv-python
# Replace with path to video file
path_to_vid = input("Enter a video file name (in ./materials folder!): ")
video = cv2.VideoCapture(f"materials/{path_to_vid}")

# Infer via the Roboflow Infer API and return the result
# Takes an httpx.AsyncClient as a parameter
async def infer(requests):
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = await requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })

    preds = resp.json()
    detections = preds['predictions']

    for bounding_box in detections:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2
        class_name = bounding_box['class']
        confidence = bounding_box['confidence']
        # position coordinates: start = (x0, y0), end = (x1, y1)
        # color = RGB-value for bounding box color, (0,0,0) is "black"
        # thickness = stroke width/thickness of bounding box
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        # draw/place bounding boxes on image
        cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=1)

        (text_width, text_height), _ = cv2.getTextSize(
            f"{class_name} | {round(float(confidence), 2)}",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)

        cv2.rectangle(img, (int(x0), int(y0)), (int(x0) + text_width, int(y0) - text_height), color=(0,0,0),
            thickness=-1)
        
        text_location = (int(x0), int(y0))
        
        cv2.putText(img, f"{class_name} | {round(float(confidence), 2)}",
                    text_location, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(255,255,255), thickness=1) 
    return img

# Main loop; infers at FRAMERATE frames per second until you press "q"
async def main():
    # Initialize
    last_frame = time.time()

    # Initialize a buffer of images
    futures = []

    async with httpx.AsyncClient() as requests:
        while 1:
            # On "q" keypress, exit
            if(cv2.waitKey(1) == ord('q')):
                break

            # Throttle to FRAMERATE fps and print actual frames per second achieved
            elapsed = time.time() - last_frame
            await asyncio.sleep(max(0, 1/FRAMERATE - elapsed))
            print((1/(time.time()-last_frame)), " fps")
            last_frame = time.time()

            # Enqueue the inference request and safe it to our buffer
            task = asyncio.create_task(infer(requests))
            futures.append(task)

            # Wait until our buffer is big enough before we start displaying results
            if len(futures) < BUFFER * FRAMERATE:
                continue

            # Remove the first image from our buffer
            # wait for it to finish loading (if necessary)
            image = await futures.pop(0)
            # And display the inference results
            cv2.imshow('image', image)

# Run our main loop
asyncio.run(main())

# Release resources when finished
video.release()
cv2.destroyAllWindows()