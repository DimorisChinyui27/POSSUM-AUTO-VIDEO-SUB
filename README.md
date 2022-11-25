

## install dependencies ###
import os
#!pip install -r requirements.txt
!pip install git+https://github.com/openai/whisper.git 
!pip install yt-dlp
!pip install moviepy --upgrade
!apt-get update
!apt install imagemagick -y


## install requirements ##
!pip install -r requirement.txt


## ImageMagic fix ##
If issues arise from ImageMagick in Linux (remove or modify the policy file):
!sudo mv /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xml.off


## VideoTranscribe ##
Change line 54 in config_defaults.py wherever moviepy library is installed. 

location of file (virtual environment)
venv\Lib\site-packages\moviepy\config_defaults.py

location of file if you are using anaconda 
It can be in C:/users/anaconda/lib/site-packages 

in windows: 
 - IMAGEMAGICK_BINARY = os.getenv('IMAGEMAGICK_BINARY', 'C:\Program Files\ImageMagick-7.0.11-Q16-HDRI\magick.exe')





## folder structure required for reading, processing and saving video files ###

- OUTPUT_FOLDER = 'outputs/'  # the final video file is created in this folder. The file name is returned by the api. 
- SOURCE_VIDEO_PATH =  'source_repository/'  ## to transcribe a video please upload the video in this folder and then call the api.
- EXPERIMENT_TEMP_FOLDER = 'experiments/'  ## used for temporary processing audio/video files



### run application ##

- The api http://127.0.0.1:5000/transcribe expects json data in the following format {'traget_lang': 'en', 'filename': "myvideo.mp4"}  "please set content type -> Content-Type: application/json 

- The api returns the output filename once completed. Use the full path to download the output video file. 

- The app support the languages as described in whisper documentation. 
Example of languages: 
    en
    de
    es
    fr


To test using curl 
!curl -X POST http://127.0.0.1:5000/transcribe -H "Content-Type: application/json" -d "{\"traget_lang\":\"en\", \"filename\":\"netherland.mp4\"}"


