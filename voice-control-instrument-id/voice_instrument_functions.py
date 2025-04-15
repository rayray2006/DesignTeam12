import time
from collections import deque
import os
from dotenv import load_dotenv
import numpy as np
import pvporcupine
import pvcobra
import whisper
from pvrecorder import PvRecorder
import torch
import string
from ultralytics import YOLO
import cv2


class Transcriber:
    def __init__(self, model) -> None:
        print("loading model")
        # TODO: put model on GPU
        self.model = whisper.load_model(model)
        print("loading model finished")
        self.prompts = os.environ.get("WHISPER_INITIAL_PROMPT", "")
        print(f"Using prompts: {self.prompts}")

    def transcribe(self, frames):
        transcribe_start = time.time()
        samples = np.array(frames, np.int16).flatten().astype(np.float32) / 32768.0

        # audio = whisper.pad_or_trim(samples)
        # print(f"{transcribe_start} transcribing {len(frames)} frames.")
        # # audio = whisper.pad_or_trim(frames)

        # # make log-Mel spectrogram and move to the same device as the model
        # mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # # decode the audio
        # options = whisper.DecodingOptions(fp16=False, language="english")
        # result = whisper.decode(self.model, mel, options)

        result = self.model.transcribe(
            audio=samples,
            language="en",
            fp16=False,
            initial_prompt=self.prompts,
        )

        # print the recognized text
        transcribe_end = time.time()
        # print(
        #     f"{transcribe_end} - {transcribe_end - transcribe_start}sec: {result.get('text')}",
        #     flush=True,
        # )
        return result.get("text", "speech not detected")

def load_voice_model():
    load_dotenv()

    porcupine = pvporcupine.create(
        access_key=os.environ.get("ACCESS_KEY"),
        keyword_paths=[os.environ.get("WAKE_WORD_MODEL_PATH")],
    )

    cobra = pvcobra.create(
        access_key=os.environ.get("ACCESS_KEY"),
    )

    recoder = PvRecorder(device_index=-1, frame_length=512)

    # frame length = 512
    # samples per frame = 16,000
    # 1 sec = 16,000 / 512

    return porcupine, cobra, recoder

def get_voice_command(porcupine, cobra, recoder):
    transcriber = Transcriber(os.environ.get("WHISPER_MODEL"))
    sample_rate = 16000
    frame_size = 512
    vad_mean_probability_sensitivity = float(os.environ.get("VAD_SENSITIVITY"))
    recoder.start()

    max_window_in_secs = 3
    window_size = sample_rate * max_window_in_secs
    samples = deque(maxlen=(window_size * 6))
    vad_samples = deque(maxlen=25)
    is_recording = False
    print("ASTRA is listening...")
    
    while True:
        data = recoder.read()
        vad_prob = cobra.process(data)
        vad_samples.append(vad_prob)
        # print(f"{vad_prob} - {np.mean(vad_samples)} - {len(vad_samples)}")
        if porcupine.process(data) >= 0:
            print(f"Detected wakeword")
            is_recording = True
            samples.clear()

        if is_recording:
            if (
                len(samples) < window_size
                or np.mean(vad_samples) >= vad_mean_probability_sensitivity
            ):
                samples.extend(data)
                # print(f"listening - samples: {len(samples)}")
            else:
                print("is_recording: False")
                command = transcriber.transcribe(samples)
                print(command)
                is_recording = False
                recoder.stop()
                porcupine.delete()
                recoder.delete()
                cobra.delete()
                return command

# Identify instrument from transcription

def get_instrument_name(command):
    # command = transcriber.transcribe(samples)
    instruments = ['forceps', 'scalpel', 'scissors', 'tweezers']
    
    instrument = ''
    for word in command.split():
        word = (word.translate(str.maketrans('', '', string.punctuation))).lower()
        if word in instruments:
            instrument = word
    if instrument == '':
        instrument = "No instrument found"
    return instrument

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
    #results.save()

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

    conf_threshold = 0.65
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