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

from oauth2client.client import GoogleCredentials

# face++
API_KEY = 'bb018c46f20c7733a9d6020f73f3725d'
API_SECRET = 'Ha0YQ-QDt8d3h5kOjxRLalxwVWpB9T23'
API_SERVER = 'http://api.us.faceplusplus.com/'

# Google Cloud Services
DEV_KEY = 'AIzaSyCy4FtXwyNm6fx6j84AUI_GM9ZYrcrEGYk '

# Source image
IMAGE_PATH = "source.jpg"

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
cv2.destroyAllWindows()

window = pyglet.window.Window()

model = prediction_service.trainedmodels()

facepp_api = facepp.API(API_KEY, API_SECRET, srv=API_SERVER)

face = facepp_api.detection.detect(img=facepp.File(IMAGE_PATH))
landmark = facepp_api.detection.landmark(face_id=face['face'][0]['face_id'], type='83p')

percentWhite = 0.0

if face['face'][0]['attribute']['race']['value'] == "White":    percentWhite = float(face['face'][0]['attribute']['race']['confidence'])/100.0
if face['face'][0]['attribute']['race']['value'] == "Black":    percentWhite = 1.43 - (float(face['face'][0]['attribute']['race']['confidence'])/100.0)
else:                                                           percentWhite = 0.85*float(face['face'][0]['attribute']['race']['confidence'])/100.0

skinTone = (255*percentWhite, 245*percentWhite, 221*percentWhite)

if face['face'][0]['attribute']['gender']['value'] == "Male":   hair = "m1"
else:                                                           hair = "f1"

#variables for face components are currently assigned. Later the variables will come from the google cloud api
eye = model.predict(project="326647037000", id="eyes", body={ "input": { "csvInstance": [value for point in getEyes(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"]
eyeb = model.predict(project="326647037000", id="eyebrow", body={ "input": { "csvInstance": [value for point in getEyebrows(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"]
nose = model.predict(project="326647037000", id="nose", body={ "input": { "csvInstance": [value for point in getNose(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"]
mouth = model.predict(project="326647037000", id="mouth", body={ "input": { "csvInstance": [value for point in getMouth(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"]
face = model.predict(project="326647037000", id="shape", body={ "input": { "csvInstance": [value for point in getShape(landmark["result"][0]["landmark"]) for value in point] } } ).execute()["outputLabel"]

hairColourStr = ""
while(hairColourStr.lower() != "black" and hairColourStr.lower() != "brown" and hairColourStr.lower() != "blonde" and hairColourStr.lower() != "red" and hairColourStr.lower() != "grey" and hairColourStr.lower() != "white"):
  
    print("What is your hair colour? (black/brown/blonde/red/grey/white): ")
    hairColourStr = str(raw_input())

hairColour = (0, 0, 0)

if hairColourStr.lower() == "brown":        hairColour = (96, 62, 39)
elif hairColourStr.lower() == "blonde":     hairColour = (255, 239, 183)
elif hairColourStr.lower() == "red":        hairColour = (204, 42, 56)
elif hairColourStr.lower() == "grey":       hairColour = (150, 150, 150)
elif hairColourStr.lower() == "white":      hairColour = (225, 225, 225)

#eye file path
if eye == "hood":               eyefp = "Components/eyes/hood.png"
elif eye == "no crease":        eyefp = "Components/eyes/o\ crease.png"
else:                           eyefp = "Components/eyes/slanted.png"

#eyebrow file path
if eyeb == "thick":             eyebfp = "Components/eyebrows/othick.png"
elif eyeb == "thin":            eyebfp = "Components/eyebrows/othin.png"
else:                           eyebfp = "Components/eyebrows/eyebrows/medium.png"

#nose file path
if nose == "in":                nosefp = "Components/onose/in.png"
else:                           nosefp = "Components/onose/out.png"

#mouth file path
if mouth == "smile":            mouthfp = "Components/mouth/smile.png"
elif mouth == "neutral":        mouthfp = "Components/mouth/oneutral.png"
else:                           mouthfp = "Components/mouth/ofrown.png"

#face file path
if face == "round":             

    facefp = "Components/oface/oround.png"
    facecfp = "Components/ofacec/oroundc.png"

elif face == "triangular":      

    facefp = "Components/oface/otriangular.png"
    facecfp = "Components/ofacec/otriangularc.png"

else:                           

    facefp = "Components/oface/square.png"
    facecfp = "Components/ofacec/squarec.png"

#hair file path
if hair == "m1":                

    hairfp = "Components/hair/m1.png"
    haircfp ="Components/hairc/m1c.png"

elif hair == "m2":              

    hairfp = "Components/hair/m2.png"
    haircfp = "Components/hairc/m2c.png"

elif hair == "f1":              

    hairfp = "Components/hair/f1.png"
    haircfp = "Components/hairc/f1c.png"

else:                           

    hairfp = "Components/hair/f2.png"
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
ifacecsprite.color = skinTone
ihairsprite = pyglet.sprite.Sprite(ihair, x=50, y=50)
ihaircsprite = pyglet.sprite.Sprite(ihairc, x=50, y=50)
ihaircsprite.color = hairColour

@window.event
def on_draw():

    window.clear()
    back.blit(1,1,1)
    ifacesprite.draw()
    ifacecsprite.draw()
    ieyesprite.draw()
    ieyebsprite.draw()
    inosesprite.draw()
    imouthsprite.draw()
    ihairsprite.draw()
    ihaircsprite.draw()

pyglet.app.run()