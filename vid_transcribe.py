


from __future__ import unicode_literals

from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
#from yt_dlp import YoutubeDL
#import yt_dlp
from IPython.display import Video
import whisper
import cv2
import pandas as pd
from moviepy.editor import VideoFileClip
import moviepy.editor as mp
from IPython.display import display, Markdown
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import os
import cv2
import uuid


## helper function used to batch the transcribtion ####
def batch_text(result, gs=32):
    """split list into small groups of group size `gs`."""
    segs = result['segments']
    length = len(segs)
    mb = length // gs
    text_batches = []
    for i in range(mb):
        text_batches.append([s['text'] for s in segs[i*gs:(i+1)*gs]])
    if mb*gs != length:
        text_batches.append([s['text'] for s in segs[mb*gs:length]])
    return text_batches


### translate a given text from source language to target language ###
def _translate(text, tokenizer, model_tr, src_lang='en', tr_lang='zh'):
    tokenizer.src_lang = src_lang
    encoded_en = tokenizer(text, return_tensors="pt", padding=True)
    generated_tokens = model_tr.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id(tr_lang))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

### translate a batch #### 
def batch_translate(texts, tokenizer, model_tr, src_lang='en', tr_lang='zh'):
    translated = []
    for t in tqdm(texts):
        tt = _translate(t, tokenizer, model_tr, src_lang=src_lang, tr_lang=tr_lang)
        translated += tt
    return translated

### translates the transcribtion to target language ###
def translate(result, tr_lang):
    ckpt = 'facebook/m2m100_418M'
    model_tr = M2M100ForConditionalGeneration.from_pretrained(ckpt)
    tokenizer = M2M100Tokenizer.from_pretrained(ckpt)


    ## create a batch of text of size 32 ###
    texts = batch_text(result, gs=32)
    texts_tr = batch_translate(texts, tokenizer, model_tr, src_lang=result['language'], tr_lang=tr_lang)

    return texts_tr




OUTPUT_FOLDER = 'outputs/'
SOURCE_VIDEO_PATH =  'source_repository/'
EXPERIMENT_TEMP_FOLDER = 'experiments/'
## create subtitle function #######
"""
    subtitle_video adds subtitle to a video given the path.

    :model_type: type of model as described in whisperai eg. tiny tiny.en medium etc
    :source_video_name: name of the input file 
    :output: name of output file
    :target_lang: target language for subtitles
    :return: describe what it returns
""" 
def subtitle_video(model_type, source_video_name, output, target_lang):
    ## First, this checks if your expermiment name is taken. If not, it will create the directory.
    ## Otherwise, we will be prompted to retry with a new name
        # Use local clip if not downloading from youtube
    import re
    
    
    try:
        exp_name = re.search('[^.]+',source_video_name)[0]
    except:
        exp_name = str(uuid.uuid4())

    ### creates a folder to save audio file for processing ##
    experiment_folder = EXPERIMENT_TEMP_FOLDER + exp_name + '/'

    try:
        os.mkdir(experiment_folder)
    except:
        print("folder exists or permission denied")

    
    audio_file = exp_name + "_audio.mp3"
    audio_file_path = experiment_folder +  audio_file
    video_file = exp_name + "_video.mp4",
    
    source_video = SOURCE_VIDEO_PATH + source_video_name

    print("video source path {}".format(source_video))
    ## read video file ##
    my_clip = mp.VideoFileClip(source_video)

    ## extract the audio from the video and save it for translation & processing ##
    my_clip.audio.write_audiofile(audio_file_path)

    # Creates an instance of whisper model using model_type variable
    model = whisper.load_model(model_type)
    

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    ## get the most probable language code ###
    source_lang_code = max(probs,key=probs.get)
    
    ## set default task to transcribe ##
    task = 'transcribe'

    ## compare the video detected language and the target language ##
    ## if the language differs translate the audio ### 
    if source_lang_code != target_lang: 
        print('changing task 2 translate')
        task = 'translate' 

    # Get text from speech for subtitles from audio file
    result = model.transcribe(audio_file_path, task = task)
    dict1 = {'start':[], 'end':[], 'text':[], 'translation':[]}

    if target_lang != "en" and task == "translate":
       txt_batches = translate(result,tr_lang=target_lang)
    else:
       txt_batches = [ item['text'] for item in result['segments'] ]

    ### creates a dictionary that contains the segment start & end positions, the corresponding text and translation 
    for orig_seg,tr_batch in zip(result['segments'],txt_batches):
        dict1['start'].append(int(orig_seg['start']))
        dict1['end'].append(int(orig_seg['end']))
        dict1['text'].append(orig_seg['text'])
        dict1['translation'].append(tr_batch)
    
    try:

            ## creates a dataframe from the above dictionary ##
            df = pd.DataFrame.from_dict(dict1)
            print(df.head())
            
            ## save the data  ### 
            df.to_csv(experiment_folder +  '/subs.csv')
            
            ## read the original video ### 
            vidcap = cv2.VideoCapture(source_video)
            success,image = vidcap.read()
            
            ## identify the video width & height 
            height = image.shape[0]
            width =image.shape[1]

            # Instantiate MoviePy subtitle generator with TextClip, subtitles, and SubtitlesClip
            generator = lambda txt: TextClip(txt, font='P052-Bold', fontsize=width/30, stroke_width=.7, color='white', stroke_color = 'black', size = (width, height*.25), method='caption')
            
            # generator = lambda txt: TextClip(txt, color='white', fontsize=20, font='Georgia-Regular',stroke_width=3, method='caption', align='south', size=video.size)
            subs = tuple(zip(tuple(zip(df['start'].values, df['end'].values)), df['translation'].values))
            subtitles = SubtitlesClip(subs, generator)
            

            # If the file was a local upload:
            video = VideoFileClip(source_video)
            ## combine the video and the subtitles ##
            final = CompositeVideoClip([video, subtitles.set_pos(('center','bottom'))])
            final.write_videofile(OUTPUT_FOLDER + '/' + output, fps=video.fps, remove_temp=True, codec="libx264", audio_codec="aac")

            return "success"
    except Exception as e: 
        return "error: {} ".format(e.__str__())

