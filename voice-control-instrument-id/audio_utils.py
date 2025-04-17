# audio_utils.py (Final Synced Version with Two Modes + Wake Word + Feedback + Background Volume Control)

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
import nltk
from nltk.corpus import words
import speech_recognition as sr
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

#nltk.download('words')

try:
    nltk.data.find("corpora/words")
except LookupError:
    nltk.download("words")

english_words = set(words.words())
custom_valid = {"astra", "scissors", "scalpel", "forceps", "needle", "give", "me", "please"}
valid_words = english_words.union(custom_valid)

def is_valid_english(word):
    return word.lower() in valid_words and word.isascii()


def install_and_import(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        __import__(import_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

for pkg in ["pydub", "librosa", "numpy", "speech_recognition", "pyttsx3"]:
    install_and_import(pkg)

if not shutil.which("ffmpeg"):
    print("FFmpeg not found. Install with: brew install ffmpeg")
    sys.exit(1)

# -------------------------------
# Globals
# -------------------------------
whisper_model = whisper.load_model("base")
recognizer = sr.Recognizer()
microphone = sr.Microphone()
wake_words = [
    "astra", "hey astra", "astraa", "austrah", "extra", "ast", "astra give", "hey astra give",
    "aster", "astro", "austro", "arstrah"
]

AUDIO_DIR = Path("/Users/charissaluk/Desktop/DT12/audio_files")
_tts_voice_id = None

# Basic whitelist (extend as needed)
VALID_TOOL_WORDS = {
    "scissors", "scalpel", "forceps", "needle", 
    "give", "me", "please", "grab", "get", "tool", "pass", "hand"
}


# -------------------------------
# Audio Utils
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

def normalize_audio(audio: AudioSegment):
    return audio.apply_gain(-audio.max_dBFS)


def identify_instruments(command, confidence_threshold=0.7):
    import difflib


    print(f"[DEBUG] Raw transcription: {command}")
    print(f"[DEBUG] Cleaned command: {command.translate(str.maketrans('', '', string.punctuation)).lower()}")

    instruments = ['forceps', 'scalpel', 'scissors', 'needle']
    alt_names = {
        'scissors': [
            'cesaurus', 'scizzards', 'sizzlers', 'sizzors', 
            'sizzers', 'sizzars', 'scissor', 'seesaws', 'sizars', 
            'scizzles', 'cicero', 'cissors', 'sizzled', 'seizer', 'cizzars'
        ],
        'forceps': [
            'four steps', 'for steps', '4 steps', 'foursteps', 'forsteps', '4steps',
            'four seps', 'for step', '4 step', 'fourstep', 'forstep', '4step',
            'fourceps', 'forceps', '4ceps', 'four ceps', 'for ceps', '4 ceps',
            'foursips', 'forsips', '4sips', 'four sips', 'for sips', '4 sips',
            'foursip', 'forsip', '4sip', 'four sip', 'for sip', '4 sip',
            'foursep', 'forsep', '4sep', 'four sep', 'for sep', '4 sep',
            'forsets', 'for sets', '4 sets', 'four sets', 'for sets', '4 sets',
            'forset', 'for set', '4 set', 'four set', 'for set', '4 set',
            'force', 'forces', 'forecepts', 'forks ups', 'forsyths', 'forcepses',
            'force apps', 'force epts', 'force eps', 'forseps', '4 seps',
            'for seps', 'four seps', 'forseize', 'fourseize', '4 seize', '4seps',
    
        ],
        'scalpel': [
            'scalpels', 'scalpel blade', 'scalp', 'scale pill', 'scalball',
            'skull pill', 'scapple', 'scalp bell', 'scowl pill', 'sculpel', 'sculptor'
        ],
        'needle': [
            'needles', 'kneadle', 'neato', 'knee doll', 'neadle',
            'knead all', 'neato', 'noodle','needle', 'kneel', 'neil'
        ]
    }

    found = {}
    # Remove punctuation and lowercase
    cleaned = command.translate(str.maketrans('', '', string.punctuation)).lower()
    words = cleaned.split()

    # Preserve original cleaned command for alias matching
    joined_text = " ".join(words)

    # Filter out non-English/irrelevant words for primary + fuzzy matching only
    filtered_words = [w for w in words if is_valid_english(w)]
    filtered_joined_text = " ".join(filtered_words)

    print(f"[DEBUG] Cleaned command: {joined_text}")
    print(f"[DEBUG] Filtered words: {filtered_words}")


        


    for inst in instruments:
        #if inst in joined_text:
        if inst in filtered_joined_text:
            found[inst] = max(found.get(inst, 0), 0.9)
    for inst, alts in alt_names.items():
        for alt in alts:
            if alt in joined_text:
                found[inst] = max(found.get(inst, 0), 0.7)
    for word in words:
        matches = difflib.get_close_matches(word, instruments, n=1, cutoff=confidence_threshold)
        if matches:
            found[matches[0]] = max(found.get(matches[0], 0), 0.7)

    for inst, conf in found.items():
        if conf >= 0.9:
            print(f"Instrument identified: {inst} (Confidence: {conf*100:.0f}%)")
        else:
            print(f"Instrument identified (Fuzzy Match): {inst} (Confidence: ~{conf*100:.0f}%)")
    print(f"[DEBUG] Final matched instruments with confidence: {found}")

    return list(found.items())


# -------------------------------
# TTS & Feedback
# -------------------------------
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

def play_feedback(keyword):
    print(f"Using AI-generated voice: Getting {keyword}")
    speak_text(f"Getting {keyword}")

def play_feedback_multiple(tool_list):
    if not tool_list:
        return
    if len(tool_list) == 1:
        play_feedback(tool_list[0])
    else:
        joined = ", then ".join(tool_list)
        print(f"Using AI-generated voice: Getting {joined}")
        speak_text(f"Getting {joined}")

def announce_test(keyword, background, volume):
    announcement = f"Starting test with {keyword} at {volume} percent volume in {background} background."
    print(announcement)
    speak_text(announcement)

# -------------------------------
# Pre-recorded Audio Processing
# -------------------------------
def process_mixed_audio_with_background_and_wakeword(
    background_path,
    voice_files_info,
    background_offset_ms=0,
    background_volume_percent=100,
    play_during_transcription=True
):
    try:
        background_wav = ensure_wav_format(background_path) if background_path else None
        temp_files = [background_wav] if background_wav else []

        if background_wav:
            full_bg = normalize_audio(AudioSegment.from_wav(background_wav)).set_frame_rate(16000).set_channels(1)
            background_audio = full_bg[background_offset_ms:]
            bg_volume_db = db_from_percentage(background_volume_percent)
            background_audio = background_audio + bg_volume_db
        else:
            background_audio = AudioSegment.silent(duration=15000)

        combined = background_audio
        for info in voice_files_info:
            voice_wav = ensure_wav_format(info["path"])
            if voice_wav != info["path"]:
                temp_files.append(voice_wav)
            voice_audio = normalize_audio(AudioSegment.from_wav(voice_wav)).set_frame_rate(16000).set_channels(1)
            voice_audio = voice_audio + db_from_percentage(info.get("volume", 100))
            combined = combined.overlay(voice_audio, position=info.get("start_ms", 0))

        max_end = max(info.get("start_ms", 0) + AudioSegment.from_wav(ensure_wav_format(info["path"])).duration_seconds * 1000 for info in voice_files_info)
        trim_point = int(max_end + 5000)
        combined = AudioSegment.silent(duration=1000) + combined[:trim_point] + AudioSegment.silent(duration=1000)

        final_mix_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        combined.export(final_mix_path, format="wav")

        if play_during_transcription:
            subprocess.run(["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", final_mix_path])

        audio_data, sr = librosa.load(final_mix_path, sr=16000)
        result = whisper_model.transcribe(audio_data.astype(np.float32), language="en", fp16=False)
        raw_transcription = result.get("text", "")
        raw_transcription = result.get("text", "")
        transcription = raw_transcription.strip().lower()
        print(f"\nTranscription result: {transcription if transcription else 'None'}")



        print(f"\nTranscription result: {transcription if transcription else 'None'}")

        wake_conf = 1.0 if any(w in transcription.lower().split() for w in wake_words) else (
            0.7 if "astra" in transcription.lower() else 0.0
        )

        instruments = identify_instruments(transcription)
        if instruments:
            tool, conf = instruments[0]
            return transcription, final_mix_path, wake_conf, tool, conf

        return transcription, final_mix_path, wake_conf, None, 0.0

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None, 0.0, None, 0.0


# -------------------------------
# Live Listening Mode
# -------------------------------
#def listen_and_transcribe_live():
def listen_and_transcribe_live(phrase_time_limit=20):
    print("\nEntering live mode. Say 'astra' to begin. Then give a command or say it together like 'astra give me scalpel'")
    speak_text("Aastra ready.")

    tools = []  # Initialize tools to avoid reference before assignment
    last_tool = None
    issued_tools = set()  # ✅ Track issued tools for cancellation
    
    while True:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Waiting...")
            audio = recognizer.listen(source, timeout=None)
        try:
            text = recognizer.recognize_google(audio).lower()
            print(f"Heard: {text}")

            if any(w in text for w in ["astra sleep", "go to sleep", "sleep", "bye astra", "buy astra", "goodbye astra", "by astra"]):
                speak_text("Are you sure you want to put Aastra to sleep?")
                try:
                    with microphone as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.3)
                        print("Listening for confirmation (yes/no)...")
                        confirm_audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)

                    confirmation = recognizer.recognize_google(confirm_audio).lower()
                    print(f"Confirmation response: {confirmation}")
                    if any(resp in confirmation for resp in ["yes", "yeah", "yup", "sure", "affirmative"]):
                        speak_text("aastra going to sleep. Bye Bye.")
                        return True
                    else:
                        speak_text("Sleep cancelled. Continuing live mode.")
                        return False

                except sr.WaitTimeoutError:
                    print("No confirmation heard. Cancelled.")
                    speak_text("No confirmation heard. Sleep cancelled.")
                    return False

                except sr.UnknownValueError:
                    print("Could not understand confirmation.")
                    speak_text("Sorry, I didn't catch that. Sleep cancelled.")
                    return False

            if any(w in text.split() for w in wake_words):
                command_text = ""  # reset

                if "cancel" in text:
                    canceled_tools = identify_instruments(text)
                    if canceled_tools:
                        for tool, _ in canceled_tools:
                            if tool in issued_tools:
                                speak_text(f"Cancelling {tool}")
                                issued_tools.remove(tool)
                            else:
                                speak_text(f"{tool} was not queued. Cannot cancel.")
                        continue
                    elif last_tool:
                        speak_text(f"Cancelling {last_tool}")
                        issued_tools.discard(last_tool)
                        continue
                    else:
                        speak_text("Nothing to cancel.")
                        continue

                # Case 1: wake + tool in same sentence
                tools = identify_instruments(text)

                # Case 2: wake only → listen again
                if not tools:
                    speak_text("Listening.")
                    retry_count = 0
                    while retry_count < 3:
                        try:
                            with microphone as source:
                                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                                print("Listening for instrument...")
                                command_audio = recognizer.listen(source, timeout=10, phrase_time_limit=phrase_time_limit)
                            command_text = recognizer.recognize_google(command_audio).lower()
                            print(f"Command heard: {command_text}")
                            tools = identify_instruments(command_text)
                            if tools:
                                break
                            retry_count += 1
                            speak_text("Sorry, I didn't catch that. Please repeat.")
                        except Exception as e:
                            print(f"Error listening for instrument: {e}")
                            retry_count += 1
                            speak_text("Something went wrong. Try again.")

                    if not tools:
                        speak_text("No valid command detected. Listening for new command.")
                        continue


                # Parse tool command
                #tools = identify_instruments(text)
                
                if "cancel" in command_text:
                    if tools:
                        for tool, _ in tools:
                            if tool == last_tool:
                                speak_text(f"Cancelling {tool}")
                                last_tool = None
                            else:
                                speak_text(f"Sorry, {tool} wasn't queued up.")
                    else:
                        speak_text("I didn’t catch what you want to cancel.")
                    continue

                
                if not tools and len(text.split()) <= 1:
                    speak_text("Listening.")
                    retry_count = 0
                    tools = []

                    while retry_count < 3 and not tools:
                        try:
                            with microphone as source:
                                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                                print("Listening for instrument...")
                                #command_audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
                                command_audio = recognizer.listen(source, timeout=15, phrase_time_limit=phrase_time_limit) # command for adaptive listening
                            command_text = recognizer.recognize_google(command_audio).lower()
                            print(f"Command heard: {command_text}")
                            if any(sleep_phrase in command_text for sleep_phrase in ["go to sleep", "astra sleep", "bye astra", "goodbye astra", "buy astra", "sleep"]):
                                speak_text("Are you sure you want to put Astra to sleep?")
                                try:
                                    with microphone as source:
                                        recognizer.adjust_for_ambient_noise(source, duration=0.3)
                                        print("Listening for confirmation (yes/no)...")
                                        confirm_audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)
                                    confirmation = recognizer.recognize_google(confirm_audio).lower()
                                    print(f"Confirmation response: {confirmation}")
                                    if any(resp in confirmation for resp in ["yes", "yeah", "yup", "sure", "affirmative"]):
                                        speak_text("Astra going to sleep. Bye Bye.")
                                        return True
                                    else:
                                        speak_text("Sleep cancelled. Continuing live mode.")
                                        return False
                                except sr.WaitTimeoutError:
                                    print("No confirmation heard. Cancelled.")
                                    speak_text("No confirmation heard. Sleep cancelled.")
                                    return False
                                except sr.UnknownValueError:
                                    print("Could not understand confirmation.")
                                    speak_text("Sorry, I didn't catch that. Sleep cancelled.")
                                    return False

                            tools = identify_instruments(command_text)
                            if not tools:
                                retry_count += 1
                                speak_text("Sorry, I didn't catch that. Please repeat.")
                        except Exception as e:
                            print(f"Error listening for instrument: {e}")
                            retry_count += 1
                            speak_text("Sorry, something went wrong. Please try again.")

                    if not tools:
                        speak_text("No valid command detected. Listening for new command.")
                        continue


            # Failsafe: Check tool confidence
                
            if tools:
                # Keep order as spoken — no sorting
                confirmed_tools = []
                low_confidence_tools = []

                for tool, conf in tools:
                    if conf >= 0.7:
                        confirmed_tools.append((tool, conf))
                    else:
                        low_confidence_tools.append((tool, conf))

                if confirmed_tools:
                    tool_names = [tool for tool, conf in confirmed_tools if tool not in issued_tools]
                    if tool_names:
                        tools_text = ", then ".join(tool_names)
                        play_feedback_multiple(tool_names)
                        issued_tools.update(tool_names)
                        last_tool = tool_names[-1]

                elif low_confidence_tools:
                    top_tool, _ = low_confidence_tools[0]
                    alt_tool = low_confidence_tools[1][0] if len(low_confidence_tools) > 1 else None
                    if alt_tool:
                        speak_text(f"Sorry, did you say {top_tool} or {alt_tool}?")
                    else:
                        speak_text(f"Did you say {top_tool}? Please confirm.")

                    try:
                        with microphone as source:
                            recognizer.adjust_for_ambient_noise(source, duration=0.3)
                            print("Listening for confirmation...")
                            # confirm_audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                            confirm_audio = recognizer.listen(source, timeout=10, phrase_time_limit=phrase_time_limit) # adaptive version


                        confirmation = recognizer.recognize_google(confirm_audio).lower()
                        print(f"Confirmation response: {confirmation}")
                        if top_tool in confirmation:
                            play_feedback(top_tool)
                            last_tool = top_tool
                        else:
                            speak_text("Command not confirmed. Cancelling.")
                    except Exception as e:
                        print(f"Could not confirm: {e}")
                        speak_text("Sorry, I didn't catch that. Cancelling.")
                else:
                    print("No tools confidently detected.")
                    speak_text("I didn't catch the instrument. Please repeat.")
        except Exception as e:
            print(f"Error recognizing speech: {e}")

