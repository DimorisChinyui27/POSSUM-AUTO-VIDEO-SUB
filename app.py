import pandas as pd
import os
import uuid

from flask import session
import os
from flask import Flask, request, redirect, sessions
import os
import datetime

import vid_transcribe as vid_app


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    
    return "hello"

@app.route("/transcribe", methods=['GET', 'POST'])
def transcribe():
    try:
        content_type = request.headers.get('Content-Type')
        print(content_type)
        if (content_type == 'application/json'):
            params = request.get_json()
            filename = params['filename'] 
            print(filename)
            target_lang = params['traget_lang']
            source_lang = 'auto'
            if 'source_lang' in params and len(params['source_lang']) > 1:
                source_lang = params['source_lang']

            output_file =  str(uuid.uuid4()) + '.mp4'
            result = vid_app.subtitle_video(
            model_type = 'tiny', # change to 'large' if you want more accurate results, 
                                #change to 'medium.en' or 'large.en' for all english language tasks,
                                #and change to 'small' or 'base' for faster inference
            output = output_file,
            source_video_name = filename,source_lang = source_lang, target_lang=target_lang)
            return output_file
    except Exception as e: 
        return "error" + str(e)
    return "done"



if __name__ == "__main__":
    app.run()


    ##production run ####
    #app.run(debug=False, host="localhost", port=int(os.environ.get("PORT", 8080)))
