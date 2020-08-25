# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:15:40 2020

@author: laksh
"""

from Package import social_distancing_configuration as config
from Package.object_detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
from flask import Flask, render_template, Response

app = Flask(__name__)

labelsPath = os.path.sep.join([config.MODEL_PATH,"coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if config.USE_GPU:
    print("[INFO] setting preferable backend and target to CUDA")
    net .setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net .setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#initaiate the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(r"pedestrians.mp4" if "pedestrians.mp4" else 0)
global writer
writer = None

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
        violate = set()
        if len(results) >= 2:
            centeroids = np.array([r[2] for r in results])
            D = dist.cdist(centeroids, centeroids, metric="euclidean")
            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < config.MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)
        #loop over the results
        for(i, (prob, bbox, centeroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centeroid
            color = (0, 255, 0)
            if i in violate:
                color = (0, 0, 255)
            cv2.rectangle(frame, (startX, startY),(endX,endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        cv2.imwrite("1.jpg",frame)
        (flag, encodedImage) = cv2.imencode(".jpg",frame)
        yield (b'--frame\r\n' b'content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')
 
@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #port = int(os.getenv('PORT', 8000))
    app.run(debug=False)
    #http_server = WSGIServer(('0.0.0.0', port), app)
    #http_server.serve_forever()
    #vs.release()
    #cv2.destroyAllWindows()
    

