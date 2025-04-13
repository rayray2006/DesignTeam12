# audio_testbench.py (Updated for Synced Modes and Cancel Handling)

import os
import numpy as np
from audio_utils import (
    process_mixed_audio_with_background_and_wakeword,
    play_feedback,
    announce_test,
    set_tts_voice,
    listen_and_transcribe_live
)
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from pydub import AudioSegment
from tqdm import tqdm
import subprocess
import time
import gc
import random
import pandas as pd
import difflib

# ------------------------------
# Configuration
# ------------------------------
AUDIO_DIR = Path("/Users/charissaluk/Desktop/DT12/audio_files")
KEYWORDS = ["scissors", "scalpel", "forceps", "needle"]
BACKGROUNDS = [
    AUDIO_DIR / "surgery_ambience_talking.mp3",
    AUDIO_DIR / "surgery_ambience_beeps.mp3",
    None
]
VOLUMES = [0, 25, 50, 75, 100]
RESULTS_DIR = Path("/Users/charissaluk/Desktop/DT12/DesignTeam12/voice-control-instrument-id/voice-control-tests/results")
PLOTS_DIR = RESULTS_DIR / "graphs"
CSV_PATH = RESULTS_DIR / "transcription_results.csv"
GAP_BETWEEN_WAKE_AND_COMMAND = 3000
GAP = 5000
DURATION_PER_PAIR = 15000

FILENAME_MAP = {
    "astra": "wakeword_female.m4a",
    "scissors": "scissors.m4a",
    "scalpel": "scalpel.m4a",
    "forceps": "forceps.m4a",
    "needle": "needle.m4a"
}

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

from itertools import product

def select_tts_voice():
    import pyttsx3
    engine = pyttsx3.init()
    voices = [v for v in engine.getProperty("voices") if "en" in str(v.languages[0]).lower() or "english" in v.name.lower()]
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
    except Exception:
        print("Invalid input. Using default voice.")

def run_tests():
    records = []
    total_tests = len(KEYWORDS) * len(BACKGROUNDS) * len(VOLUMES)
    progress = tqdm(total=total_tests, ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ({percentage:.0f}%)")

    start_time_dict = {}
    for bg in BACKGROUNDS:
        if bg is None:
            start_time_dict["none"] = 0
        else:
            bg_audio = AudioSegment.from_file(bg)
            bg_duration = len(bg_audio) / 1000
            max_start = max(0, int((bg_duration - 60) * 1000))
            start_time_dict[bg.stem] = random.randint(0, max_start)

    for keyword, bg, vol in product(KEYWORDS, BACKGROUNDS, VOLUMES):
        command_path = AUDIO_DIR / FILENAME_MAP[keyword]
        wakeword_path = AUDIO_DIR / FILENAME_MAP["astra"]
        bg_name = "none" if bg is None else bg.stem

        print(f"Testing {keyword} with {bg_name} at {vol}% volume")
        announce_test(keyword, bg_name, vol)

        voice_files_info = []
        offset = 0
        voice_files_info.append({
            "path": wakeword_path,
            "volume": 100,
            "start_ms": offset + 1000
        })
        voice_files_info.append({
            "path": command_path,
            "volume": 100,
            "start_ms": offset + 1000 + GAP_BETWEEN_WAKE_AND_COMMAND
        })

        transcription, mix_path, wake_confidence, command_name, command_confidence = process_mixed_audio_with_background_and_wakeword(
            background_path=str(bg) if bg else None,
            voice_files_info=voice_files_info,
            background_offset_ms=start_time_dict[bg_name] if bg_name in start_time_dict else 0,
            background_volume_percent=vol,  # <-- ADD THIS LINE
            play_during_transcription=True
        )


        if command_name and command_confidence >= 0.7:
            play_feedback(command_name)

        records.append({
            "tool": keyword,
            "background": bg_name,
            "volume": vol,
            "transcription": transcription or "",
            "wakeword_confidence": wake_confidence,
            "command_detected": int(command_name == keyword),
            "command_confidence": command_confidence
        })

        progress.update(1)
        time.sleep(1)
        gc.collect()

    progress.close()

    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False)
    print(f"Results saved to: {CSV_PATH}")
    return df

def plot_background_vs_accuracy(df):
    for keyword in KEYWORDS:
        plt.figure()
        for bg_name in df['background'].unique():
            sub = df[(df['tool'] == keyword) & (df['background'] == bg_name)]
            grouped = sub.groupby('volume')['command_detected'].mean()
            plt.plot(grouped.index, grouped.values, marker='o', label=bg_name)
        plt.title(f"Accuracy vs Volume for '{keyword}'")
        plt.xlabel("Background Volume (%)")
        plt.ylabel("Detection Accuracy")
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid(True)
        plt.savefig(PLOTS_DIR / f"accuracy_vs_volume_{keyword}.png")
        plt.close()

def plot_keyword_accuracy_by_background(df):
    for bg_name in df['background'].unique():
        plt.figure()
        sub = df[df['background'] == bg_name]
        grouped = sub.groupby('tool')['command_detected'].mean()
        plt.bar(grouped.index, grouped.values)
        plt.title(f"Keyword Accuracy in Background: {bg_name}")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(axis='y')
        plt.savefig(PLOTS_DIR / f"keyword_accuracy_{bg_name}.png")
        plt.close()

def plot_confidence_breakdown(df):
    pass

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    select_tts_voice()
    print("\nSelect Mode:")
    print("1 - Run full test suite with audio files")
    print("2 - Live voice command test")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "2":
        while True:
            done = listen_and_transcribe_live()
            if done:
                break
    else:
        df_results = run_tests()
        plot_background_vs_accuracy(df_results)
        plot_keyword_accuracy_by_background(df_results)
        print("\nAll tests complete. Graphs saved to:", PLOTS_DIR)
