######## Video Stream Object Detection Using Tensorflow-trained Classifier #########
#
#Code used from multiple places 
#Author: Adam Whitman
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
import serial
import RPi.GPIO as GPIO


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/

#################################################################################################################################################
class VideoStream:
    """Camera object that controls video streaming"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture('rtsp:admin:Password02@@@192.168.2.64:554/Streaming/Channels/101/?transportmode=unicast')
        #self.stream = cv2.VideoCapture("los_angeles.mp4")
        #amcrest rtsp link
        #self.stream = cv2.VideoCapture('rtsp://admin:Password02@@@192.168.1.158:554/cam/realmonitor?channel=1&subtype=0')
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
parser.add_argument('--streamurl', help='The full URL of the video stream e.g. http://ipaddress:port/stream/video.mjpeg',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
STREAM_URL = args.streamurl
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

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

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

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


#########################################Added Code########################################
#Motion detection using means of gaussian 2(fancy tracking)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=700,varThreshold=50, detectShadows=True)




############################FOR SENDING TEXT MESSAGES
ser = serial.Serial("/dev/ttyS0",115200)
ser.flushInput()

phone_number = '15012869821' #********** change it to the phone number you want to call
#text_message = 'Intruder detected'
power_key = 6
rec_buff = ''

def send_at(command,back,timeout):
	rec_buff = ''
	ser.write((command+'\r\n').encode())
	time.sleep(timeout)
	if ser.inWaiting():
		time.sleep(0.01 )
		rec_buff = ser.read(ser.inWaiting())
	if back not in rec_buff.decode():
		print(command + ' ERROR')
		print(command + ' back:\t' + rec_buff.decode())
		return 0
	else:
		print(rec_buff.decode())
		return 1

def SendShortMessage(phone_number,text_message):
	
	print("Setting SMS mode...")
	send_at("AT+CMGF=1","OK",1)
	#print("Sending Short Message")
	answer = send_at("AT+CMGS=\""+phone_number+"\"",">",2)
	ser.write(text_message.encode())
	ser.write(b'\x1A')
	print("Should be sent")
	answer = send_at('','OK',20)


##Differnt motion tracking
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,4))
kernal = np.ones((5,5),np.uint(8))
background = None



#For tracking a person
count = 0

#Prevents Errors from print
center_points_prev_frame = []

###For tracking the object
tracking_objects = {}

track_id = 0
text_count = 0

tracked_object_types ={"person","car","truck"}
min_distance = {"car":5, "person":5, "truck":5 }
max_distance = {"car":500, "person":500, "truck":500 }
min_contour = {"car":150, "truck":150, "person":150}

text_sent_id={} 



##############################################################CAMERA LOOP
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    
    
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    
    # Grab frame from video stream
    frame1 = videostream.read()
    frame2 = videostream.read()
    
        
    ##array for center points
    center_points_cur_frame=[]
    
    #array for tracking tests
    
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
    
    ##Draws bounding box around motion for simple motion tracking#
    #for c in cnts:
    #    if cv2.contourArea(c) <1000:
    #        continue
    #    (x, y, w, h) = cv2.boundingRect(c)
    #    cv2.rectangle(frame,(x, y), (x + w, y + h),(0,255,255),3)
    #    
    #cv2.imshow("contours", frame)
    
  
   
   ###############################Fancy Motion Tracking (resource HEAVY#####
   # Motion detection 
   # mask = object_detector.apply(frame)
   # #Sets the threshold on the mask to white also ignores shadows
   # _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)   
   #
   # _ , contours, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   #      
   # #To see the mask applied#
   # #cv2.imshow("Mask",mask)#            
   ###############################
   
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
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index

            if object_name in tracked_object_types:

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
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
    
    
   
   #tracking center point of bounding boxes 
    for pt, i in center_points_cur_frame:
         cv2.circle(frame, (pt), 5, (0,255,255), -1)
    
         
    
    
    count =1
        
    if count < 0:
         for pt, i in center_points_cur_frame:
             for pt2, i in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 75:
                    tracking_objects[track_id] = pt
                    text_sent_id[text_id] = pt
                    #track_id += 1
                    
                
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()
        text_sent_id_copy = text_sent_id.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
                                  
            for (pt, i) in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                print(distance,object_id,track_id)
                
                

                

                # Update IDs position
                if distance < 75:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if (pt,i) in center_points_cur_frame:
                        center_points_cur_frame.remove((pt,i))                    
                    continue
            
            if not object_exists:
                tracking_objects.pop(object_id)
                if track_id in text_sent_id:
                    text_sent_id.pop(object_id)
                    
                    
                      
        #tracking_objects_copy = tracking_objects.copy()
        #center_points_cur_frame_copy = center_points_cur_frame.copy()
        
        for (pt, i) in center_points_cur_frame_copy:
            for object_id, pt2 in tracking_objects_copy.items():
               object_exists = False         
               for (pt, i) in center_points_cur_frame_copy:
                    object_name = labels[int(classes[i])]
                    if object_name in tracked_object_types:
                        
                        distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                        #print("Distance is = ",distance, track_id)
                        if distance >= min_distance[object_name] and distance <= max_distance[object_name]:
                            #print(object_name," found ",track_id, "is moving by distance")
                            for c in cnts:
                                
                                if cv2.contourArea(c) > min_contour[object_name]:
                                    #future conditional if
                                    M = cv2.moments(c)
                                    c_cx = int(M['m01']/M['m00'])
                                    c_cy = int(M['m01']/M['m00'])
                                    #print(c_cx, c_cy, track_id)
                                    distance2 = math.hypot(c_cx - pt[0], c_cy - pt[1])
                                    #distance2 = math.hypot(c_cx - cx, c_cy - cy)
                                    
                                    #print(distance2," distance 2 =")
                                    if distance2 >= 15 and distance2 <= 1900:
                                        #print(distance2, "distance 2")
                                        #loop to send message for each intruder detected only once
                                        if not track_id in text_sent_id:
                                            text_message = ("Intruder detected with motion")
                                            print(object_name,"detected with motion")
                                            
                                            #Sends text message
                                            #SendShortMessage(phone_number,text_message)
                                            text_sent_id[text_count] = track_id
                                            text_count += 1                                                                
                                    
                        
                
                #for cnt in contours:
        #calculate area of small contours
                    #area = cv2.contourArea(cnt)
                    
          
         #Updates for new id's found 
        
        for pt, i in center_points_cur_frame:
            tracking_objects[track_id] = pt
            object_name = labels[int(classes[i])]
            #print(object_name," found ",track_id, " for new id found")
            track_id += 1
            #print(track_id)
            #count = 0                    
    
    
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 255, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 255, 255), 2)
        
    
    
    
    
    
    
    #### For the fancy motion tracking    
   # ### Calculates area of contour/where text messages will reside
   # #for cnt in contours:
   #     #calculate area of small contours and remove them from detection
   #  #   area = cv2.contourArea(cnt)
   #     #if area > 200:
   #         
   #         #draws actual contour around motion
   #         #cv2.drawContours(frame, [cnt], -1, (0,255,255), 1)
   #         
   #         ##draws bounding boxes around motion
   #        # x, y, w, h = cv2.boundingRect(cnt)
   #        # cv2.rectangle(frame,(x,y),(x + w,y + h),(0,255,255),3)
        
      
    
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

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
cv2.destroyAllWindows()
videostream.stop()
