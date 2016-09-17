import re

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
IMAGE_PATH = "C:/Users/Richard/Desktop/me.jpg"

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

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord(" "):
        break

cv2.imwrite(IMAGE_PATH, frame)

model = prediction_service.trainedmodels()

print(model.insert(project="326647037000", body={"storageDataLocation": "face_data_20160917/eye_data.csv", "modelType": "classification", "id": "eyes"}).execute())

facepp_api = facepp.API(API_KEY, API_SECRET, srv=API_SERVER)

face = facepp_api.detection.detect(img=facepp.File(IMAGE_PATH))
landmark = facepp_api.detection.landmark(face_id=face['face'][0]['face_id'], type='83p')

window = pyglet.window.Window()

vertex_list = pyglet.graphics.vertex_list(len(getShape(landmark["result"][0]["landmark"])), ('v2f', [value for point in getShape(landmark["result"][0]["landmark"]) for value in point]))

@window.event
def on_draw():
    window.clear()
    vertex_list.draw(pyglet.gl.GL_POINTS)

pyglet.app.run()


cv2.destroyAllWindows()
