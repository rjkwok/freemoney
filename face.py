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
IMAGE_PATH = "me.jpg"

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
cv2.destroyAllWindows()

window = pyglet.window.Window()

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
	rgb = image.get_region(int(colormark['result'][0]['landmark']['right_eyebrow_upper_middle']['x'] + random.uniform(-1.0, 1.0)*radius), int(colormark['result'][0]['landmark']['right_eyebrow_upper_middle']['y'] + random.uniform(-1.0, 1.0)*radius), 1, 1).get_image_data().get_data("RGB", 3)
	rgb_vals.append((ord(rgb[0]), ord(rgb[1]), ord(rgb[2]))) 

skinTone = (sum([v[0] for v in rgb_vals]) / float(len(rgb_vals)), sum([v[1] for v in rgb_vals]) / float(len(rgb_vals)), sum([v[2] for v in rgb_vals]) / float(len(rgb_vals)))

if skinTone[0] < 80 or skinTone[1] < 80 or skinTone[2] < 80: skinTone = (80, 80, 80)

#variables for face components are currently assigned. Later the variables will come from the google cloud api
eye = model.predict(project="326647037000", id="eyes", body={ "input": { "csvInstance": [value for point in getEyes(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"]
eyeb = model.predict(project="326647037000", id="eyebrow", body={ "input": { "csvInstance": [value for point in getEyebrows(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"]
nose = model.predict(project="326647037000", id="nose", body={ "input": { "csvInstance": [value for point in getNose(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"]
mouth = model.predict(project="326647037000", id="mouth", body={ "input": { "csvInstance": [value for point in getMouth(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"]
face = model.predict(project="326647037000", id="shape", body={ "input": { "csvInstance": [value for point in getShape(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"]
facec = "squarec"
hair = "m1"
hairc = "m1c"

#file paths
eyefp = ""
eyebfp = ""
nosefp = ""
mouthfp = "" 
facefp = ""
hairfp = ""
haircfp = ""

#eye file path
if eye == "hood":
    eyefp = "Components/eyes/hood.png"
elif eye == "no crease":
    eyefp = "Components/eyes/o\ crease.png"
else:
    eyefp = "Components/eyes/slanted.png"
#eyebrow file path
if eyeb == "thick":
    eyebfp = "Components/eyebrows/othick.png"
elif eyeb == "thin":
    eyebfp = "Components/eyebrows/othin.png"
else:
    eyebfp = "Components/eyebrows/eyebrows/medium.png"
#nose file path
if nose == "in":
    nosefp = "Components/onose/in.png"
else:
    nosefp = "Components/onose/out.png"
#mouth file path
if mouth == "smile":
    mouthfp = "Components/mouth/smile.png"
elif mouth == "neutral":
    mouthfp = "Components/mouth/oneutral.png"
else:
    mouthfp = "Components/mouth/ofrown.png"

#face file path
if face == "round":
    facefp = "Components/oface/oround.png"
elif face == "triangular":
    facefp = "Components/oface/otriangular.png"
else:
    facefp = "Components/oface/square.png"


#facec file path
if facec == "roundc":
    facecfp = "Components/ofacec/oroundc.png"
elif facec == "triangularc":
    facecfp = "Components/ofacec/otriangularc.png"
else:
    facecfp = "Components/ofacec/squarec.png"

#hair file path
if hair == "m1":
    hairfp = "Components/hair/m1.png"
elif hair == "m2":
    hairfp = "Components/hair/m2.png"

elif hair == "f1":
    hairfp = "Components/hair/f1.png"
else:
    hairfp = "Components/hair/m1.png"

#hairc file path
if hairc == "m1c":
    haircfp ="Components/hairc/m1c.png"
elif hairc == "m2c":
    haircfp = "Components/hairc/m2c.png"
elif hairc == "f1c":
    haircfp = "Components/hairc/f1c.png"
else:
    haircfp = "Components/hairc/f2c.png"


ieye = pyglet.image.load(eyefp)
ieyeb = pyglet.image.load(eyebfp)
inose = pyglet.image.load(nosefp)
imouth = pyglet.image.load(mouthfp)
iface = pyglet.image.load(facefp)
ifacec = pyglet.image.load(facecfp)
ihair = pyglet.image.load(hairfp)
ihairc = pyglet.image.load(haircfp)
back = pyglet.image.load("Components/white-background-300x246.png")


ieyesprite = pyglet.sprite.Sprite(ieye, x=50, y=50)
ieyebsprite = pyglet.sprite.Sprite(ieyeb, x=50, y=50)
inosesprite = pyglet.sprite.Sprite(inose, x=50, y=50)
imouthsprite = pyglet.sprite.Sprite(imouth, x=50, y=50)
ifacesprite = pyglet.sprite.Sprite(iface, x=50, y=50)
ifacecsprite = pyglet.sprite.Sprite(ifacec, x=50, y=50)
ihairsprite = pyglet.sprite.Sprite(ihair, x=50, y=50)
ihaircsprite = pyglet.sprite.Sprite(ihairc, x=50, y=50)


@window.event
def on_draw():
    window.clear()
    #yglet.gl.glClearColor(1, 1, 1, 1)

    back.blit(1,1,1)
    #ieye.blit(0,0,0)
    #ieyeb.blit(0,0,0)
    #inose.blit(0,0,0)
    #imouth.blit(0,0,0)
    #iface.blit(0,0,0)

    #back.draw()
    #ieyesprite.color = (255,0,0)
    #ieyebsprite.color = (255,0,0)

    ifacecsprite.color = skinTone
    #ifacesprite.color = (255,0,0)
    ifacesprite.draw()
    ifacecsprite.draw()
    #print(ifacesprite.color)
    


    ieyesprite.draw()
    ieyebsprite.draw()

    inosesprite.draw()
    imouthsprite.draw()
    ihairsprite.draw()
    ihaircsprite.draw()
    #ieyeb.draw()
    
    #vertex_list_eyes.draw(pyglet.gl.GL_POINTS)
    ##vertex_list_nose.draw(pyglet.gl.GL_POINTS)
    ##vertex_list_eyebrows.draw(pyglet.gl.GL_POINTS)
    ##vertex_list_mouth.draw(pyglet.gl.GL_POINTS)

pyglet.app.run()