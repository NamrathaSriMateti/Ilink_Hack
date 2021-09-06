# Ilink_Hack

Title: Crowd and Queue monitoring using Computer Vision

Description: Monitoring crowd and flagging if social distancing is not maintained in the queue.

Technologies Used: Python, HTML, Flask Framework, YoloV3, Computer Vision

Guide:

1. To get started, Install the required libraries using requirements.txt
2. Download yolov3.weights using this link 'https://pjreddie.com/media/files/yolov3.weights' and save it in models folder.
3. To run the web - application, use 'python app.py' command. 
4. On the home page, you get three options: My Own Camera
										                        Live stream Video
										                        Video on my PC


5. On selecting "Video on my PC" option, you can upload the video from your device. 
  { The other two options are the future scope of this application which can be implemented to extend the functionality. By selecting the "My Own Camera" option, you can connect the webcam of your laptop or desktop to get the input video for detecting social distancing. For "Live Stream Video", you can connect an external camera via IP address for the required video. }

6. Once the required input video has been uploaded in MP4 format, click the "Detect Social Distancing" button. The model will detect the persons not following the social distancing and flag them in red bounding boxes. 
The output of our input video can be found in the base directory with the name "Output_thres_4_60" in AVI format.

7. The code uses yolov3 with OpenCV which has been customised to detect people in the video. The distance between two persons is detected with pixel count(MIN_DISTANCE) which can be configured in the code based on the camera's angle and distance from the crowd.

8. The Yolo weights, coco names and config used in the model can be found in the 'models' folder.











