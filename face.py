import re
import time
import random

import pyglet
import facepp
import apiclient
import oauth2client
from googleapiclient.discovery import build

import numpy as np
import cv2

# face++
API_KEY = 'bb018c46f20c7733a9d6020f73f3725d'
API_SECRET = 'Ha0YQ-QDt8d3h5kOjxRLalxwVWpB9T23'
API_SERVER = 'http://api.us.faceplusplus.com/'

# Google Cloud Services
DEV_KEY = 'AIzaSyCy4FtXwyNm6fx6j84AUI_GM9ZYrcrEGYk '

# Source image
IMAGE_PATH = "/Users/2943644/Desktop/obama.jpeg"

def getEyes(landmarks):

	return [(landmarks[key]['x'], landmarks[key]['y']) for key in landmarks if re.search("_eye_", key)]

def getEyebrows(landmarks):

	return [(landmarks[key]['x'], landmarks[key]['y']) for key in landmarks if re.search("eyebrow", key)]

def getNose(landmarks):

	return [(landmarks[key]['x'], landmarks[key]['y']) for key in landmarks if re.search("nose", key)]

def getMouth(landmarks):

	return [(landmarks[key]['x'], landmarks[key]['y']) for key in landmarks if re.search("mouth", key)]

def getShape(landmarks):

	return [(landmarks[key]['x'], landmarks[key]['y']) for key in landmarks if re.match("contour", key)]

#############################################################
# Main execution
#############################################################

#
# Brows: "thick", "medium", "thin"
# Eyes: "hood", "no crease", "slanted"
# Nose: "in", "out"
# Mouth: "smile", "neutral", "frown"
# Shape: "square", "triangular", "round"
#

from oauth2client.client import GoogleCredentials
credentials = GoogleCredentials.get_application_default()

cap = cv2.VideoCapture(0)
prediction_service = build('prediction', 'v1.6', developerKey=DEV_KEY, credentials=credentials)

frame = None
while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    #frame = cv2.imread(IMAGE_PATH, 0)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord(" "):
        break

cv2.imwrite(IMAGE_PATH, frame)

model = prediction_service.trainedmodels()

facepp_api = facepp.API(API_KEY, API_SECRET, srv=API_SERVER)

face = facepp_api.detection.detect(img=facepp.File(IMAGE_PATH))
landmark = facepp_api.detection.landmark(face_id=face['face'][0]['face_id'], type='83p')

colormark = facepp_api.detection.landmark(face_id=face['face'][0]['face_id'])

image = pyglet.image.load(IMAGE_PATH)

random.seed()

rgb_vals = []


radius = 20
for i in range(100):
	rgb = image.get_region(int(colormark['result'][0]['landmark']['nose_tip']['x'] + random.uniform(-1.0, 1.0)*radius), int(colormark['result'][0]['landmark']['nose_tip']['y'] + random.uniform(-1.0, 1.0)*radius), 1, 1).get_image_data().get_data("RGB", 3)
	rgb_vals.append((ord(rgb[0]), ord(rgb[1]), ord(rgb[2]))) 

skinTone = (sum([v[0] for v in rgb_vals]) / float(len(rgb_vals)), sum([v[1] for v in rgb_vals]) / float(len(rgb_vals)), sum([v[2] for v in rgb_vals]) / float(len(rgb_vals)))
print(skinTone)

print(model.predict(project="326647037000", id="eyes", body={ "input": { "csvInstance": [value for point in getEyes(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"])
print(model.predict(project="326647037000", id="mouth", body={ "input": { "csvInstance": [value for point in getMouth(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"])
print(model.predict(project="326647037000", id="eyebrow", body={ "input": { "csvInstance": [value for point in getEyebrows(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"])
print(model.predict(project="326647037000", id="nose", body={ "input": { "csvInstance": [value for point in getNose(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"])
print(model.predict(project="326647037000", id="shape", body={ "input": { "csvInstance": [value for point in getShape(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"])

cv2.destroyAllWindows()
 