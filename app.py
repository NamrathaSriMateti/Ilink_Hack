from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

from flask import Flask, render_template, request, redirect


app = Flask(__name__)

#base path to YOLO directory
MODEL_PATH="models"

#Initializing minimum probability to filter weak detections when applying non-maxima suppression
MIN_CONF=0.3
NMS_THRESH=0.4

#the minimun safe distance (in pixels) between people to maintain social distancing
MIN_DISTANCE=60

@app.route('/')
def upload():
    return render_template("home.html")

def detect(boundary, network, ln, personIdx=0):
    (H, W) = boundary.shape[:2]
    results = []
    blob = cv2.dnn.blobFromImage(boundary, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    network.setInput(blob)
    layerOutputs = network.forward(ln)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores) # extracting the class ID
            confidence = scores[classID]  # extracting the probability
            if classID == personIdx and confidence > MIN_CONF:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    return results

@app.route('/social_distance_detector',methods=['POST'])
def social_distance_detector():
    if request.method == 'POST':
        f = request.files['file']
        fn = f.filename
        print('fn:',fn)
        print('f.filename',f.filename)
        f.save(fn)
        result = request.form
        dstype = result['dstype']
        print('dstype:',dstype)
        if dstype == 'camera':
            own_camera = cv2.VideoCapture(0)

            while(True):

                ret, boundary = own_camera.read()
                cv2.imshow('boundary', boundary)
                
                #to stop the code, press 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            own_camera.release()
            cv2.destroyAllWindows()
            return 'OK'
        elif dstype == 'live':
            return "You can input Ip Address of CC camera as [capture = cv2.VideoCapture('Ip Address')]"
        
        elif dstype == 'pc':
            ap = argparse.ArgumentParser()
            ap.add_argument("-i", "--input", type=str, default=fn,
                help="path to (optional) input video file")
            output_file = fn.split('.')[0] +'_1'+ '.avi'
            print('output_file:',output_file)
            ap.add_argument("-o", "--output", type=str, default=output_file,
                help="path to (optional) output video file")
            ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not output boundary should be displayed")
            args = vars(ap.parse_args())

            labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
            LABELS = open(labelsPath).read().strip().split("\n")
            weightsPath = os.path.sep.join([MODEL_PATH, "yolov3.weights"])
            configPath = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])
            network = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

            ln = network.getLayerNames()
            ln = [ln[i[0] - 1] for i in network.getUnconnectedOutLayers()]           
            vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
            writer = None

            while True:
                (grabbed, boundary) = vs.read()
                if not grabbed:
                    break
                boundary = imutils.resize(boundary, width=700)
                results = detect(boundary, network, ln,
                    personIdx=LABELS.index("person"))

                violate = set()
                total = set()
                if len(results) >= 2:
                    centroids = np.array([r[2] for r in results])
                    D = dist.cdist(centroids, centroids, metric="euclidean")
                    for i in range(0, D.shape[0]):
                        for j in range(i + 1, D.shape[1]):
                            if D[i, j] < MIN_DISTANCE:
                                violate.add(i)
                                violate.add(j)
                            else:
                                total.add(i)
                                total.add(j)

                for (i, (prob, bbox, centroid)) in enumerate(results):
                    (startX, startY, endX, endY) = bbox
                    (cX, cY) = centroid

                    if i in violate:
                        color = (0, 0, 255)

                        cv2.rectangle(boundary, (startX, startY), (endX, endY), color, 2)
                        cv2.circle(boundary, (cX, cY), 5, color, 1)

                text = "People violating Social Distancing : {}".format(len(violate))
                
                cv2.putText(boundary, text, (10, boundary.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)

                if args["display"] > 0:
                    cv2.imshow("boundary", boundary)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                if args["output"] != "" and writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(args["output"], fourcc, 25,
                        (boundary.shape[1], boundary.shape[0]), True)

                if writer is not None:
                    writer.write(boundary)
        return 'Output video is Genereated, Please check the application folder'
        
if __name__ == '__main__':
    #app.debug = True
    app.run(host='127.0.0.1', port=5000)
