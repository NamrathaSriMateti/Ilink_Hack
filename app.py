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

def detect(frame, net, ln, personIdx=0):
    # storing dimensions of the frame and  initialize the list of results
    (H, W) = frame.shape[:2]
    results = []
    
    #constructing a video from the input video by detecting people using bounding boxes and probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    centroids = []
    confidences = []

    # looping over each of the layer outputs
    for output in layerOutputs:
        # looping over each of the detections
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores) # extracting the class ID
            confidence = scores[classID]  # extracting the probability

            # filter detections by (1) ensuring that the object
            # detected was a person and (2) that the minimum
            # confidence is met
            if classID == personIdx and confidence > MIN_CONF:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # return the list of results
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

                # Capture the video frame
                # by frame
                ret, frame = own_camera.read()

                # Display the resulting frame
                cv2.imshow('frame', frame)
                
                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # After the loop release the cap object
            own_camera.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
            return 'OK'
        elif dstype == 'live':
            return "You can input Ip Address of CC camera as [capture = cv2.VideoCapture('Ip Address')]"
        
        elif dstype == 'pc':
        
            # construct the argument parse and parse the arguments
            ap = argparse.ArgumentParser()
            ap.add_argument("-i", "--input", type=str, default=fn,
                help="path to (optional) input video file")
            output_file = fn.split('.')[0] +'_1'+ '.avi'
            print('output_file:',output_file)
            ap.add_argument("-o", "--output", type=str, default=output_file,
                help="path to (optional) output video file")
            ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not output frame should be displayed")
            args = vars(ap.parse_args())

            # load the COCO class labels our YOLO model was trained on
            labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
            LABELS = open(labelsPath).read().strip().split("\n")
            
            # derive the paths to the YOLO weights and model configuration
            weightsPath = os.path.sep.join([MODEL_PATH, "yolov3.weights"])
            configPath = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])

            # load our YOLO object detector trained on COCO dataset (80 classes)
            print("[INFO] loading YOLO from disk...")
            net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

            # determine only the *output* layer names that we need from YOLO
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            # initialize the video stream and pointer to output video file
            print("[INFO] accessing video stream...")
            
            vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
            writer = None

            # loop over the frames from the video stream
            while True:
                # read the next frame from the file
                (grabbed, frame) = vs.read()

                # if the frame was not grabbed, then we have reached the end
                # of the stream
                if not grabbed:
                    break

                # resize the frame and then detect people (and only people) in it
                frame = imutils.resize(frame, width=700)
                results = detect(frame, net, ln,
                    personIdx=LABELS.index("person"))

                # initialize the set of indexes that violate the minimum social
                # distance
                violate = set()
                total = set()
                # ensure there are *at least* two people detections (required in
                # order to compute our pairwise distance maps)
                if len(results) >= 2:
                    # extract all centroids from the results and compute the
                    # Euclidean distances between all pairs of the centroids
                    centroids = np.array([r[2] for r in results])
                    D = dist.cdist(centroids, centroids, metric="euclidean")

                    # loop over the upper triangular of the distance matrix
                    for i in range(0, D.shape[0]):
                        for j in range(i + 1, D.shape[1]):
                            # check to see if the distance between any two
                            # centroid pairs is less than the configured number
                            # of pixels
                            if D[i, j] < MIN_DISTANCE:
                                # update our violation set with the indexes of
                                # the centroid pairs
                                violate.add(i)
                                violate.add(j)
                            else:
                                total.add(i)
                                total.add(j)

                # loop over the results
                for (i, (prob, bbox, centroid)) in enumerate(results):
                    # extract the bounding box and centroid coordinates, then
                    # initialize the color of the annotation
                    (startX, startY, endX, endY) = bbox
                    (cX, cY) = centroid
                    #color = (0, 255, 0)

                    # if the index pair exists within the violation set, then
                    # update the color
                    if i in violate:
                        color = (0, 0, 255)

                    # draw (1) a bounding box around the person and (2) the
                    # centroid coordinates of the person,
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        cv2.circle(frame, (cX, cY), 5, color, 1)

                # draw the total number of social distancing violations on the
                # output frame
                text = "People violating Social Distancing : {}".format(len(violate))
                
                cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
                

                # check to see if the output frame should be displayed to our
                # screen
                if args["display"] > 0:
                    # show the output frame
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF

                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break

                # if an output video file path has been supplied and the video
                # writer has not been initialized, do so now
                if args["output"] != "" and writer is None:
                    # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(args["output"], fourcc, 25,
                        (frame.shape[1], frame.shape[0]), True)

                # if the video writer is not None, write the frame to the output
                # video file
                if writer is not None:
                    writer.write(frame)
        return 'Output video is Genereated, Please check the application folder'


if __name__ == '__main__':
    #app.debug = True
    app.run(host='127.0.0.1', port=5000)
