# audio_utils.py (Refactored with Live Listening Support)

import os
import sys
import shutil
import tempfile
import subprocess
import string
import gc
import numpy as np
import librosa
import whisper
import random
import speech_recognition as sr
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

# -------------------------------
# Install & Import Requirements
# -------------------------------
def install_and_import(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        __import__(import_name)
    except ModuleNotFoundError:
        print(f"{package_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} installed successfully.")

for pkg in ["pydub", "librosa", "numpy", "speech_recognition", "pyttsx3"]:
    install_and_import(pkg)

# -------------------------------
# Ensure FFmpeg is Available
# -------------------------------
def check_ffmpeg_installed():
    if not shutil.which("ffmpeg"):
        print("\nFFmpeg is not installed or not in your PATH.")
        print("Install it using Homebrew: brew install ffmpeg\n")
        sys.exit(1)
    else:
        print("FFmpeg is available.")

check_ffmpeg_installed()

# -------------------------------
# Global Model & Recognizers
# -------------------------------
whisper_model = whisper.load_model("base")
recognizer = sr.Recognizer()
microphone = sr.Microphone()
wake_words = ["astra", "aster", "astro"]
AUDIO_DIR = Path("/Users/charissaluk/Desktop/DT12/audio_files")

# -------------------------------
# Utility Functions
# -------------------------------
def ensure_wav_format(input_path):
    input_path = str(input_path)
    if input_path.lower().endswith(".wav"):
        return input_path
    audio = AudioSegment.from_file(input_path)
    temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav_file.name, format="wav")
    return temp_wav_file.name

def db_from_percentage(volume_percent):
    return np.interp(volume_percent, [0, 100], [-60, 0])

def identify_instruments(command, confidence_threshold=0.7):
    import difflib
    instruments = ['forceps', 'scalpel', 'scissors', 'needle']
    alt_names = {
        'scissors': ['cesaurus', 'scizzards'],
        'forceps': ['four steps', 'for seps', 'four step'],
        'scalpel': [],
        'needle': []
    }
    found = {}
    cleaned = command.translate(str.maketrans('', '', string.punctuation)).lower()
    words = cleaned.split()
    joined_text = " ".join(words)

    for inst in instruments:
        if inst in words:
            found[inst] = max(found.get(inst, 0), 1.0)

    for inst, alts in alt_names.items():
        for alt in alts:
            if alt in joined_text:
                found[inst] = max(found.get(inst, 0), 0.7)

    for word in words:
        matches = difflib.get_close_matches(word, instruments, n=1, cutoff=confidence_threshold)
        if matches:
            found[matches[0]] = max(found.get(matches[0], 0), 0.7)

    for inst, conf in found.items():
        if conf == 1.0:
            print(f"Instrument identified: {inst} (Confidence: 100%)")
        elif conf == 0.7:
            print(f"Instrument identified (Fuzzy Match): {inst} (Confidence: ~70%)")
    return list(found.items())

# -------------------------------
# TTS Voice Configuration
# -------------------------------
_tts_voice_id = None

def set_tts_voice(voice_id):
    global _tts_voice_id
    _tts_voice_id = voice_id

def speak_text(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        if _tts_voice_id:
            engine.setProperty('voice', _tts_voice_id)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS failed: {e}")

def announce_test(keyword, background, volume):
    announcement = f"Starting test with {keyword} at {volume} percent volume in {background} background."
    print(announcement)
    speak_text(announcement)

# -------------------------------
# Feedback Audio or TTS
# -------------------------------
def play_feedback(keyword):
    print(f"Using AI-generated voice: Getting {keyword}")
    speak_text(f"Getting {keyword}")

# -------------------------------
# Real-time Listening Mode
# -------------------------------
def listen_and_transcribe_live():
    print("Entering live mode. Say 'astra sleep' to exit.")
    current_tool = None

    while True:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Waiting for wake word...")
            try:
                audio = recognizer.listen(source, timeout=None)
                query = recognizer.recognize_google(audio)
                print(f"Heard: {query}")
                query = query.lower()

                if any(exit_cmd in query for exit_cmd in ["astra sleep", "go to sleep", "sleep"]):
                    speak_text("Are you sure you want to put Astra to sleep?")
                    print("Confirmation requested: Astra sleep")
                    try:
                        confirmation_audio = recognizer.listen(source, timeout=10)
                        confirmation_text = recognizer.recognize_google(confirmation_audio).lower()
                        print(f"Confirmation response: {confirmation_text}")
                        if any(word in confirmation_text for word in ["yes", "yeah", "yep", "affirmative", "sure"]):
                            print("Confirmed. Exiting live mode.")
                            speak_text("Okay. Putting Astra to sleep.")
                            break
                        else:
                            print("Sleep cancelled. Continuing live mode.")
                            speak_text("Okay, continuing.")
                            continue
                    except sr.WaitTimeoutError:
                        print("No confirmation heard. Continuing.")
                        speak_text("No confirmation heard. Continuing.")
                        continue

                if any(w in query for w in wake_words):
                    command = query
                    for w in wake_words:
                        command = command.replace(w, "").strip()

                    if "cancel" in command:
                        if current_tool:
                            speak_text(f"{current_tool} cancelled")
                            print(f"Command cancelled: {current_tool}")
                            current_tool = None
                        else:
                            speak_text("No tool to cancel")
                        continue

                    if command:
                        print("Detected command in same phrase.")
                        command_query = command
                    else:
                        speak_text("Wake word detected. Awaiting command.")
                        print("Wake word confirmed. Listening for instrument (up to 30s)...")
                        try:
                            audio = recognizer.listen(source, timeout=30)
                            command_query = recognizer.recognize_google(audio)
                            print(f"Command heard: {command_query}")
                        except sr.WaitTimeoutError:
                            print("Timeout: No command detected in 30s window.")
                            continue

                    if "cancel" in command_query:
                        if current_tool:
                            speak_text(f"{current_tool} cancelled")
                            print(f"Command cancelled: {current_tool}")
                            current_tool = None
                        else:
                            speak_text("No tool to cancel")
                        continue

                    instruments = identify_instruments(command_query)
                    spoken = set()
                    for tool, confidence in instruments:
                        if confidence >= 0.7 and tool not in spoken:
                            current_tool = tool
                            play_feedback(tool)
                            spoken.add(tool)
                    if not spoken:
                        print("Tool confidence too low or not detected. No feedback played.")

            except sr.WaitTimeoutError:
                print("Timeout: No speech detected.")
            except Exception as e:
                print(f"Recognition error: {e}")
