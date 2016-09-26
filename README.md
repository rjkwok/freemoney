#BanterFace
## Inspiration
In the show "Community" an episode includes the characters playing a video game. The game scans their faces and creates RPG-styled sprites to play in the game. 
## What it does
BanterFace uses the Face++ API to locate key points on the user's face, such as the tip of their nose and their pupils. Face++ also finds important characteristics about the User in the photo such as mood, age, race, etc. Google Cloud's API in machine learning is trained by many test photos to be able to identify key characteristics in the face that Face++ could not such as eyelid shape, eyebrow thickness, nose shape, and jaw structure. A new sprite of the user is made based on the information received from Google Cloud's API.
## How We built it
We built BanterFace using Google Cloud API, Face++ API, and pyglet libraries.
## Challenges We ran into
Matching the skin tones of the sprites to the User's image was difficult and still is not very accurate. 
We did not have enough time to implement a GUI.
## Accomplishments that We're proud of
We're very pleased that we have a functioning project that involves machine learning and facial recognition - two complex topics. 
## What we learned
We are all happy to experience machine learning and implementing it. Implementing Face++ and working with it is also new to us. In addition, most of us did not have experience with pyglet prior to this hack.
## Sample Portrait
[Absolute README link](http://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/000/411/154/datas/gallery.jpg)
