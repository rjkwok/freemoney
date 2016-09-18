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

prediction_service = build('prediction', 'v1.6', developerKey=DEV_KEY)

prediction_service.hostedmodels()

facepp_api = facepp.API(API_KEY, API_SECRET, srv=API_SERVER)

face = facepp_api.detection.detect(img=facepp.File(IMAGE_PATH))
landmark = facepp_api.detection.landmark(face_id=face['face'][0]['face_id'], type='83p')

window = pyglet.window.Window()


vertex_list_eyes = pyglet.graphics.vertex_list(len(getEyes(landmark["result"][0]["landmark"])), ('v2f', [value for point in getEyes(landmark["result"][0]["landmark"]) for value in point]))
vertex_list_nose = pyglet.graphics.vertex_list(len(getNose(landmark["result"][0]["landmark"])), ('v2f', [value for point in getNose(landmark["result"][0]["landmark"]) for value in point]))
vertex_list_eyebrows = pyglet.graphics.vertex_list(len(getEyebrows(landmark["result"][0]["landmark"])), ('v2f', [value for point in getEyebrows(landmark["result"][0]["landmark"]) for value in point]))
vertex_list_mouth = pyglet.graphics.vertex_list(len(getMouth(landmark["result"][0]["landmark"])), ('v2f', [value for point in getMouth(landmark["result"][0]["landmark"]) for value in point]))

#variables for face components are currently assigned. Later the variables will come from the google cloud api
eye = "hood"
eyeb = "thick"
nose = "in"
mouth = "smile"
face = "square"
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
	eyefp = "Components2/eyes/hood.png"
elif eye == "no crease":
	eyefp = "Components2/eyes/o\ crease.png"
else:
	eyefp = "Components2/eyes/slanted.png"
#eyebrow file path
if eyeb == "thick":
	eyebfp = "Components2/eyebrows/othick.png"
elif eyeb == "thin":
	eyebfp = "Components2/eyebrows/othin.png"
else:
	eyebfp = "Components2/eyebrows/eyebrows/medium.png"
#nose file path
if nose == "in":
	nosefp = "Components2/onose/in.png"
else:
	nosefp = "Components2/onose/on.png"
#mouth file path
if mouth == "smile":
	mouthfp = "Components2/mouth/smile.png"
elif mouth == "neutral":
	mouthfp = "Components2/mouth/oneutral.png"
else:
	mouthfp = "Components2/mouth/ofrown.png"

#face file path
if face == "round":
	facefp = "Components2/oface/oround.png"
elif face == "triangular":
	facefp = "Components2/oface/otriangular.png"
else:
	facefp = "Components2/oface/square.png"


#facec file path
if facec == "roundc":
	facecfp = "Components2/ofacec/oroundc.png"
elif facec == "triangularc":
	facecfp = "Components2/ofacec/otriangularc.png"
else:
	facecfp = "Components2/ofacec/squarec.png"

#hair file path
if hair == "m1":
	hairfp = "Components2/hair/m1.png"
elif hair == "m2":
	hairfp = "Components2/hair/m2.png"

elif hair == "f1":
	hairfp = "Components2/hair/f1.png"
else:
	hairfp = "Components2/hair/m1.png"

#hairc file path
if hairc == "m1c":
	haircfp ="Components2/hairc/m1c.png"
elif hairc == "m2c":
	haircfp = "Components2/hairc/m2c.png"
elif hairc == "f1c":
	haircfp = "Components2/hairc/f1c.png"
else:
	haircfp = "Components2/hairc/f2c.png"






ieye = pyglet.image.load(eyefp)
ieyeb = pyglet.image.load(eyebfp)
inose = pyglet.image.load(nosefp)
imouth = pyglet.image.load(mouthfp)
iface = pyglet.image.load(facefp)
ifacec = pyglet.image.load(facecfp)
ihair = pyglet.image.load(hairfp)
ihairc = pyglet.image.load(haircfp)
back = pyglet.image.SolidColorImagePattern((255, 255, 255, 255))


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
    ihairsprite.draw()
    ihaircsprite.draw()
    ifacecsprite.color = (0,0,255)
    print(ifacecsprite.color)
    #ifacesprite.color = (255,0,0)
    ifacesprite.draw()
    ifacecsprite.draw()
    #print(ifacesprite.color)
    


    ieyesprite.draw()
    ieyebsprite.draw()

    inosesprite.draw()
    imouthsprite.draw()

    #ieyeb.draw()
    
    #vertex_list_eyes.draw(pyglet.gl.GL_POINTS)
    ##vertex_list_nose.draw(pyglet.gl.GL_POINTS)
    ##vertex_list_eyebrows.draw(pyglet.gl.GL_POINTS)
    ##vertex_list_mouth.draw(pyglet.gl.GL_POINTS)

pyglet.app.run()

