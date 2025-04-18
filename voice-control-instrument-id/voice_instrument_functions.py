import time
from collections import deque
import os
import numpy as np
import whisper
import torch
import string
from ultralytics import YOLO
import cv2
from audio_utils import (
    listen_and_transcribe_live,
    set_tts_voice,
    speak_text
)

def choose_tts_voice():
    import pyttsx3
    engine = pyttsx3.init()
    print(engine)
    # voices = [v for v in engine.getProperty("voices") if "en" in str(v.languages[0]).lower() or "english" in v.name.lower()]
    # this line gets rid of index out of bounds error
    voices = [v for v in engine.getProperty("voices") if (v.languages and "en" in str(v.languages[0]).lower()) or "english" in v.name.lower()]

    print("\nAvailable English TTS Voices:")
    for i, voice in enumerate(voices):
        print(f"[{i}] {voice.name} ({voice.id})")
    try:
        choice = int(input("\nSelect a voice by number (e.g., 0 for default): "))
        if 0 <= choice < len(voices):
            set_tts_voice(voices[choice].id)
            print(f"Selected voice: {voices[choice].name}\n")
        else:
            print("Invalid choice. Using default voice.")
    except:
        print("Invalid input. Using default voice.")
    

# Identify instrument from transcription

def get_instrument_name():
    adaptive_phrase_time_limit = 10
    done = False
    while True:
        try:
            # run this if statement BEFORE listen_and_transcribe_live
            # so it breaks right after finding an instrument
            if done:
                break
            done, instruments = listen_and_transcribe_live(phrase_time_limit=adaptive_phrase_time_limit)

        except TypeError:
            # Backward compatibility fallback
            done = listen_and_transcribe_live()
            if done:
                break
    return instruments

# Yolo trained on DocCheck (Rona) dataset

def get_camera_img(pipeline):
    img_name = "inst-camera.png"
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.poll_for_frames()
    # depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("Failed to get frames")
    else:
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(img_name, color_image)
    
    return img_name

def load_model(path, reload):
    # load yolov5 from online
    # TODO: change to local
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=reload)  # load a custom model
    return model

def identify_instrument(model, img_path, instrument):
    results = model(img_path)  # predict on an image
    

    # Translate results
    results.names = {0: 'Standard Anatomical Tweezers',
     1: 'Slim Anatomical Tweezers',
     2: 'Surgical Tweezers',
     3: 'Splinter Tweezers',
     4: 'Scalpel Handle No. 3',
     5: 'Scalpel Handle No. 4',
     6: 'Clenched Scalpel',
     7: 'Narrow Scalpel',
     8: 'Surgical Scissors Sharp/Sharp',
     9: 'Surgical Scissors Sharp/Narrow',
     10: 'Standard Dissecting Scissors',
     11: 'Dissecting Needle'}
    
    #### use basic instrument names (For design day poster)
    # results.names = {0: 'Tweezers',
    #  1: 'Tweezers',
    #  2: 'Tweezers',
    #  3: 'Tweezers',
    #  4: 'Scalpel',
    #  5: 'Scalpel',
    #  6: 'Scalpel',
    #  7: 'Scalpel',
    #  8: 'Scissors',
    #  9: 'Scissors',
    #  10: 'Scissors',
    #  11: 'Needle'}
    # results.save()
    
    # map labels to basic voice commands "forceps", "scalpel", "scissors", "needle"
    map_instruments = {results.names[0]: 'forceps',
                       results.names[1]: 'forceps',
                       results.names[2]: 'forceps',
                       results.names[3]: 'forceps',
                       results.names[4]: 'scalpel',
                       results.names[5]: 'scalpel',
                       results.names[6]: 'scalpel',
                       results.names[7]: 'scalpel',
                       results.names[8]: 'scissors',
                       results.names[9]: 'scissors',
                       results.names[10]: 'scissors',
                       results.names[11]: 'needle'}

    detections = results.xyxy[0]

    conf_threshold = 0.10
    # instruments = 'forceps'
    
    # TODO: account for when there's multiple types of the same instrument
    
    x_midpoint = 0
    y_midpoint = 0
    
    for *box, conf, cls in detections:
        cls = int(cls)
        x1, y1, x2, y2 = box
        # if detection matches the instrument from voice command
        if map_instruments[results.names[cls]] == instrument and conf >= conf_threshold :
            x_midpoint = (x1 + x2) / 2
            y_midpoint = (y1 + y2) / 2
            print(f"Box center: ({x_midpoint:.0f}, {y_midpoint:.0f}) | Confidence: {conf:.2f} | Class: {results.names[cls]}")
            break
    
    if x_midpoint == 0:
        print("No instrument found")

    return x_midpoint, y_midpoint