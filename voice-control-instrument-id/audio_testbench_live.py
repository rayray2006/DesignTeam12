# audio_testbench_live.py (Menu + Background Playback + Adaptive Listening + Live Transcription)

import os
import time
import threading
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play
from audio_utils import (
    listen_and_transcribe_live,
    set_tts_voice,
    speak_text
)

# -----------------------------
# Configurations
# -----------------------------
AUDIO_DIR = Path("voice-control-instrument-id")
BACKGROUNDS = [
    AUDIO_DIR / "surgery_ambience_talking.mp3",
    AUDIO_DIR / "surgery_ambience_beeps.mp3",
    None
]

# -----------------------------
# Background Playback
# -----------------------------
def start_background_playback():
    def play_loop():
        bg_file = os.environ.get("ASTRA_BG_FILE")
        volume = int(os.environ.get("ASTRA_BG_VOLUME", 100))
        if not bg_file:
            return
        while True:
            try:
                audio = AudioSegment.from_file(bg_file)
                gain = np.interp(volume, [0, 100], [-60, 0])
                audio = audio.set_frame_rate(16000).set_channels(1).apply_gain(gain)
                play(audio)
            except Exception as e:
                print(f"Background playback error: {e}")
                break

    threading.Thread(target=play_loop, daemon=True).start()

# -----------------------------
# Menu Utility
# -----------------------------
def choose_background():
    print("\nSelect background track:")
    for i, bg in enumerate(BACKGROUNDS):
        name = "None (Silent)" if bg is None else bg.name
        print(f"[{i}] {name}")
    choice = input("Enter choice number: ").strip()
    try:
        choice = int(choice)
        if 0 <= choice < len(BACKGROUNDS):
            return BACKGROUNDS[choice]
    except:
        pass
    print("Invalid selection. Defaulting to silent background.")
    return None

def choose_volume():
    print("\nSelect background volume (0–100):")
    try:
        volume = int(input("Enter volume: ").strip())
        return max(0, min(100, volume))
    except:
        print("Invalid input. Using 100% volume.")
        return 100

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

# -----------------------------
# Main Entry
# -----------------------------
if __name__ == "__main__":
    choose_tts_voice()
    bg_path = choose_background()
    bg_volume = choose_volume()

    if bg_path:
        os.environ["ASTRA_BG_FILE"] = str(bg_path)
        os.environ["ASTRA_BG_VOLUME"] = str(bg_volume)
        print(f"\nLaunching live mode with background: {bg_path.name} at {bg_volume}% volume\n")
        start_background_playback()
    else:
        print("\nLaunching live mode with silent background\n")

    adaptive_phrase_time_limit = 10

    done = False
    while True:
        try:
            # run this if statement BEFORE listen_and_transcribe_live
            # so it breaks right after finding an instrument
            if done:
                break
            done = listen_and_transcribe_live(phrase_time_limit=adaptive_phrase_time_limit)
            
        except TypeError:
            # Backward compatibility fallback
            done = listen_and_transcribe_live()
            if done:
                break