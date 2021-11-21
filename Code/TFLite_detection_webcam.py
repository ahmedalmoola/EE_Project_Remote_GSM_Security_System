######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import math

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.6)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--phone', help='Get phone number to change receiver number',
                    default='phone.txt')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
PHONE_NUMBER = args.phone
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
TERMINATE = 'terminate.txt'
DETECT = 'detect.txt'

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to phone.txt, which contains the phone number of receiver
PATH_TO_NUMBER = os.path.join(CWD_PATH,MODEL_NAME,PHONE_NUMBER)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Path to terminal program file
PATH_TO_TERMINATE = os.path.join(CWD_PATH,MODEL_NAME,TERMINATE)

PATH_TO_DETECT = os.path.join(CWD_PATH,MODEL_NAME,DETECT)

# # Load phone number
# with open(PATH_TO_NUMBER, 'r') as f1:
#     phonenumber = [line.strip() for line in f1.readlines()]
# number = phonenumber[0]

# Load the command to close system
with open(PATH_TO_TERMINATE, 'w') as file:
    file.write("No \n")
# with open(PATH_TO_TERMINATE, 'r') as file1:
#     file2 = [line.strip() for line in file1.readlines()]
# command = file2[0]
# print(command)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

##Differnt motion tracking
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,4))
kernal = np.ones((5,5),np.uint(8))
background = None

#Prevents Errors from print
center_points_prev_frame = []

###For tracking the object
## original tracking objects library
tracking_objects = {}
#keeps name of object type
tracking_object_name = {}
#used to calc distqnce
tracking_object_distance = {}
#used for keeping object alive if it steps off of frame
tracking_object_keep_alive = {}
#how many frames to keep alive
KEEP_ALIVE_NUM_START = 30

track_id = 0
text_count = 0

tracked_object_types ={"person","car","truck"}
min_distance = {"car":5, "person":5, "truck":5 }
max_distance = {"car":800, "person":800, "truck":800 }
min_contour = {"car":150, "truck":150, "person":150}

text_sent_id={} 
last_text_time = 0

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()
    
    # Point current frame
    center_points_cur_frame = []

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    
    ###Less resource intensive motion tracking##
    if background is None:
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = cv2.GaussianBlur(background, (21,21),0)
        continue
   
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21,21),0)
    
    diff = cv2.absdiff(background, gray_frame)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff,es, iterations=2)
    
    _, cnts,_ = cv2.findContours(diff.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    
    with open(PATH_TO_NUMBER, 'r') as f1:
        phonenumber = [line.strip() for line in f1.readlines()]
    number = phonenumber[0]
    #print(number)
    
    with open(PATH_TO_TERMINATE, 'r') as file1:
        file2 = [line.strip() for line in file1.readlines()]
    command = file2[0]
    if command == "yes":
        break
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # Look up object name from "labels" array using class index
            object_name = labels[int(classes[i])]
            if object_name in tracked_object_types:

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
            
                #print(center_points_cur_frame)
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                
                #Gets center point of bounding box
                cx = int((xmin + xmax)/2)
                cy = int((ymin + ymax)/2)
                center_points_cur_frame.append(((cx,cy),i))
                #print("cx = ",cx,"cy = ", cy)
                
    #tracking center point of bounding boxes 
    for pt, i in center_points_cur_frame:
         cv2.circle(frame, (pt), 5, (0,255,255), -1)
    
    
    ##
    tracking_objects_copy = tracking_objects.copy()
    center_points_cur_frame_copy = center_points_cur_frame.copy()
    text_sent_id_copy = text_sent_id.copy()
    for (object_id, pt2) in tracking_objects_copy.items():
        object_exists = False
        for (pt,i) in center_points_cur_frame_copy:
            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
            
            # Update IDs position
            if distance < 200:
                tracking_objects[object_id] = pt
                tracking_object_name[object_id] = labels[int(classes[i])]
                tracking_object_distance[object_id] = distance
                tracking_object_keep_alive[object_id] = KEEP_ALIVE_NUM_START
                object_exists = True
                if (pt,i) in center_points_cur_frame:
                    center_points_cur_frame.remove((pt,i))
                continue

        # Remove IDs lost
        if not object_exists:
            if tracking_object_keep_alive[object_id] > 0:
                tracking_object_keep_alive[object_id] = tracking_object_keep_alive[object_id] - 1
            elif tracking_object_keep_alive[object_id] == 0:
                tracking_objects.pop(object_id)
                tracking_object_name.pop(object_id)
                tracking_object_distance.pop(object_id)
                tracking_object_keep_alive.pop(object_id)
                    
    duplicated_found = True
    
    while(duplicated_found == True):
        tracking_objects_copy = tracking_objects.copy()
        duplicated_found = False
        for object_id, pt in tracking_objects_copy.items():
            if duplicated_found:
                break
            for object_id_b, pt2 in tracking_objects_copy.items():
                if object_id != object_id_b:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                    if distance < 5:
                        duplicated_found = True
                        object_id_to_remove = max(object_id, object_id_b)
                        tracking_objects.pop(object_id_to_remove)
                        tracking_object_name.pop(object_id_to_remove)
                        tracking_object_distance.pop(object_id_to_remove)
                        tracking_object_keep_alive.pop(object_id_to_remove)
                        break
                    
    ##For sending texts
    tracking_objects_copy = tracking_objects.copy()
    for object_id, pt2 in tracking_objects_copy.items():
        object_exists = False
        #print(type(tracked_object_types))
        object_name = tracking_object_name[object_id]
        if object_name in tracked_object_types:
            #distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
            distance = tracking_object_distance[object_id]
            #print(distance)
            #distance = math.sqrt(+pow((pt2[0] - pt[0]),2)+pow((pt2[1] - pt[1]),2))
            #print("Distance is = ",distance, track_id)
            if distance >= min_distance[object_name] and distance <= max_distance[object_name]:
                #print("Intruderdistcheck--", object_name," found ",track_id, "is moving by distance")
                for c in cnts:
                        
                    if cv2.contourArea(c) > min_contour[object_name]:
                        #future conditional if
                        M = cv2.moments(c)
                        c_cx = int(M['m10']/M['m00'])
                        c_cy = int(M['m01']/M['m00'])
                        #print(c_cx, c_cy, track_id)
                        distance2 = math.hypot(c_cx - pt[0], c_cy - pt[1])
                        #distance2 = math.hypot(c_cx - cx, c_cy - cy)
                            
                            
                        if distance2 >= 45 and distance2 <= 200:
                                
                            #loop to send message for each intruder detected only once
                            cur_time = time.time()

                            if not object_id in text_sent_id and (tracking_object_keep_alive[object_id] == KEEP_ALIVE_NUM_START) and (cur_time - last_text_time)> 60.0:
                                last_text_time = cur_time
                                #print("Intruderdistcheck--ID[",object_id,"]", distance2," distance 2 =")
                                text_message = ("Intruder detected with motion")
                                if object_name == "person":
                                    print(object_name,"Intruderdistcheck-- detected with motion with ID", object_id)
                                    with open(PATH_TO_DETECT, 'w') as file:
                                        file.write("person")
                                elif object_name == "car" or object_name == "truck":
                                    print(object_name,"Vehicle -- detected with motion with ID", object_id)
                                #Sends text message
                                #SendShortMessage(phone_number,text_message)
                                text_sent_id[text_count] = object_id
                                text_count += 1                                                                
                  
    # Add new IDs found
    for (pt,i) in center_points_cur_frame:
        tracking_objects[track_id] = pt
        tracking_object_name[track_id] = labels[int(classes[i])]
        tracking_object_distance[track_id] = 0
        tracking_object_keep_alive[track_id] = KEEP_ALIVE_NUM_START
        object_name = labels[int(classes[i])]
        #print(object_name," found ",track_id, " for new id found")
        track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 255, 255), -1)
        cv2.putText(frame, str(object_id)+" " + str(tracking_object_keep_alive[object_id])+"/"+str(KEEP_ALIVE_NUM_START), (pt[0], pt[1] - 7), 0, 1, (0, 255, 255), 2)
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
#     print("Tracking objects")
#     print(tracking_objects, track_id)
# 
#     print("CUR FRAME LEFT PTS")
#     print(center_points_cur_frame)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
print("System is terminated")
cv2.destroyAllWindows()
videostream.stop()
