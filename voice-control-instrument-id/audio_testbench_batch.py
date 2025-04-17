import os
import csv
import random
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from statistics import mean, stdev
from pydub import AudioSegment
from audio_utils import process_mixed_audio_with_background_and_wakeword

# Paths
AUDIO_DIR = Path("/Users/charissaluk/Desktop/DT12/audio_files")
RESULTS_DIR = Path("/Users/charissaluk/Desktop/DT12/DesignTeam12/voice-control-instrument-id/voice-control-tests/results")
PLOTS_DIR = RESULTS_DIR / "graphs"
CSV_PATH = RESULTS_DIR / "transcription_results.csv"
EXCEL_PATH = RESULTS_DIR / "transcription_results.xlsx"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
KEYWORDS = ["scissors", "scalpel", "forceps", "needle"]
VOLUMES = [0, 25, 50, 75, 100]
GENDERS = ["female", "male"]
BACKGROUNDS = [
    AUDIO_DIR / "surgery_ambience_talking.mp3",
    AUDIO_DIR / "surgery_ambience_beeps.mp3",
]
N_RUNS = 5
SEGMENT_DURATION_MS = 15000
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

FILENAME_MAP = {
    "astra_female": AUDIO_DIR / "astra_female_2.m4a",
    "astra_male": AUDIO_DIR / "astra_male.m4a",
    "scissors_female": AUDIO_DIR / "scissors_female_2.m4a",
    "scissors_male": AUDIO_DIR / "scissors_male.m4a",
    "scalpel_female": AUDIO_DIR / "scalpel_female_2.m4a",
    "scalpel_male": AUDIO_DIR / "scalpel_male.m4a",
    "forceps_female": AUDIO_DIR / "forceps_female_2.m4a",
    "forceps_male": AUDIO_DIR / "forceps_male.m4a",
    "needle_female": AUDIO_DIR / "needle_female_2.m4a",
    "needle_male": AUDIO_DIR / "needle_male.m4a",
    "cancel_female": AUDIO_DIR / "cancel_female_2.m4a",
    "cancel_male": AUDIO_DIR / "cancel_male.m4a",
    "sleep_female": AUDIO_DIR / "sleep_female_2.m4a",
    "sleep_male": AUDIO_DIR / "sleep_male.m4a",
}

def get_random_start_ms(audio_path, segment_duration_ms):
    audio = AudioSegment.from_file(audio_path)
    max_start = len(audio) - segment_duration_ms - 1000
    return random.randint(0, max_start)

# Run tests
aggregated_results = []

for bg in BACKGROUNDS:
    bg_name = bg.stem
    for gender in GENDERS:
        for run_idx in range(N_RUNS):
            # One shared offset for all volumes and keywords in this run
            bg_offset_ms = get_random_start_ms(bg, SEGMENT_DURATION_MS)
            bg_offset_min = bg_offset_ms // 60000
            bg_offset_sec = (bg_offset_ms % 60000) // 1000
            offset_str = f"{bg_offset_min:02d}:{bg_offset_sec:02d}"
            print(f"\n=== Run {run_idx+1}/{N_RUNS} - {bg_name} | {gender} | BG Offset = {offset_str} ===")

            for keyword in KEYWORDS:
                for volume in VOLUMES:
                    print(f"🔍 {keyword.upper()} @ {volume}% Volume")

                    voice_files = [
                        {"path": FILENAME_MAP[f"astra_{gender}"], "start_ms": 0, "volume": 100},
                        {"path": FILENAME_MAP[f"{keyword}_{gender}"], "start_ms": 3000, "volume": 100},
                    ]

                    transcription, _, wake_conf, detected_tool, tool_conf = process_mixed_audio_with_background_and_wakeword(
                        background_path=bg,
                        voice_files_info=voice_files,
                        background_volume_percent=volume,
                        background_offset_ms=bg_offset_ms,
                        play_during_transcription=False
                    )

                    aggregated_results.append({
                        "background": bg_name,
                        "volume": volume,
                        "gender": gender,
                        "keyword": keyword,
                        "run": run_idx + 1,
                        "wakeword_detected": 1 if wake_conf >= 0.7 else 0,
                        "tool_detected": 1 if detected_tool == keyword else 0,
                        "tool_confidence": tool_conf,
                        "transcription": transcription,
                        "bg_offset_ms": bg_offset_ms,
                        "bg_offset_timestamp": offset_str
                    })

# Save to CSV and Excel
df = pd.DataFrame(aggregated_results)
df.to_csv(CSV_PATH, index=False)
df.to_excel(EXCEL_PATH, index=False)

# Summary stats
summary = df.groupby(["background", "volume", "gender", "keyword"]).agg({
    "wakeword_detected": ["mean", "std"],
    "tool_detected": ["mean", "std"]
}).reset_index()

summary.columns = ["background", "volume", "gender", "keyword",
                   "wakeword_accuracy", "wakeword_stdev",
                   "tool_accuracy", "tool_stdev"]

# Plot results
for bg in summary["background"].unique():
    for gender in summary["gender"].unique():
        subset = summary[(summary["background"] == bg) & (summary["gender"] == gender)]
        grouped = subset.groupby("volume").agg({
            "wakeword_accuracy": "mean",
            "wakeword_stdev": "mean",
            "tool_accuracy": "mean",
            "tool_stdev": "mean"
        }).reset_index()

        plt.figure()
        plt.errorbar(grouped["volume"], grouped["wakeword_accuracy"], yerr=grouped["wakeword_stdev"], label="Wakeword", fmt='o-')
        plt.errorbar(grouped["volume"], grouped["tool_accuracy"], yerr=grouped["tool_stdev"], label="Instrument", fmt='x--')
        plt.title(f"Accuracy with Std Dev - {bg} - {gender}")
        plt.xlabel("Background Volume (%)")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
        plt.savefig(PLOTS_DIR / f"accuracy_stddev_{bg}_{gender}.png")
        plt.close()
