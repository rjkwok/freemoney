import pyglet
import facepp
import apiclient
import oauth2client
from googleapiclient.discovery import build

# face++
API_KEY = 'bb018c46f20c7733a9d6020f73f3725d'
API_SECRET = 'Ha0YQ-QDt8d3h5kOjxRLalxwVWpB9T23'
API_SERVER = 'http://api.us.faceplusplus.com/'

# Google Cloud Services
DEV_KEY = 'AIzaSyCy4FtXwyNm6fx6j84AUI_GM9ZYrcrEGYk '

# Source image
IMAGE_PATH = "C:/Users/Richard/Desktop/me.jpg"

#############################################################
# Main execution
#############################################################

prediction_service = build('prediction', 'v1.6', developerKey=DEV_KEY)

prediction_service.hostedmodels()

facepp_api = facepp.API(API_KEY, API_SECRET, srv=API_SERVER)

face = facepp_api.detection.detect(img=facepp.File(IMAGE_PATH))

print("{0} {1}".format(face['face'][0]['attribute']['race']['value'], face['face'][0]['attribute']['gender']['value']))
