import os
import csv
import random
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from statistics import mean, stdev
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import random
from audio_utils import process_mixed_audio_with_background_and_wakeword
import time

print("Script started")

# Updated paths
AUDIO_DIR = Path("/Users/charissaluk/Desktop/DT12/audio_files")
RESULTS_DIR = Path("/Users/charissaluk/Desktop/DT12/DesignTeam12/voice-control-instrument-id/voice-control-tests/results")
PLOTS_DIR = RESULTS_DIR / "graphs"
CSV_PATH = RESULTS_DIR / "transcription_results.csv"
EXCEL_PATH = RESULTS_DIR / "transcription_results.xlsx"
LOG_PATH = RESULTS_DIR / "transcription_log.txt"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
KEYWORDS = ["scissors", "scalpel", "forceps", "needle"]
VOLUMES = [100]
GENDERS = ["female", "male"]
BACKGROUNDS = [AUDIO_DIR / "surgery_ambience_talking.mp3", None]
N_RUNS = 40 #min 35 to get statistically significant results
SEGMENT_DURATION_MS = 15000
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

FILENAME_MAP = {
    f"{kw}_{gender}": AUDIO_DIR / f"{kw}_{gender}.m4a"
    for kw in KEYWORDS + ["astra"]
    for gender in GENDERS
}

def get_random_start_ms(audio_path, segment_duration_ms):
    if audio_path is None:
        return 0
    audio = AudioSegment.silent(duration=30000)
    max_start = len(audio) - segment_duration_ms - 1000
    return random.randint(0, max_start)

aggregated_results = []
log_lines = []

for bg in BACKGROUNDS:
    bg_name = bg.stem if bg else "no_background"
    print(f"\nProcessing background: {bg_name}")
    for gender in GENDERS:
        for run_idx in range(N_RUNS):
            bg_offset_ms = get_random_start_ms(bg, SEGMENT_DURATION_MS)
            offset_str = f"{bg_offset_ms // 60000:02d}:{(bg_offset_ms % 60000) // 1000:02d}"
            for keyword in KEYWORDS:
                for volume in VOLUMES:
                    print(f"  → Gender: {gender} | Run: {run_idx+1} | Keyword: {keyword}")
                    voice_files = [
                        {"path": FILENAME_MAP[f"astra_{gender}"], "start_ms": 0, "volume": 100},
                        {"path": FILENAME_MAP[f"{keyword}_{gender}"], "start_ms": 3000, "volume": 100},
                    ]
                    transcription, _, wake_conf, detected_tool, tool_conf = process_mixed_audio_with_background_and_wakeword(
                        background_path=bg,
                        voice_files_info=voice_files,
                        background_volume_percent=volume,
                        background_offset_ms=bg_offset_ms,
                        play_during_transcription=False # set true to play audio
                    )

                    row = {
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
                    }

                    aggregated_results.append(row)
                    log_lines.append(f"[{bg_name.upper()}] ({gender}) Run {run_idx+1:02d} - {keyword}: "
                                     f"{'✔️' if row['tool_detected'] else 'none_detected'} | Transcription: {transcription}")
                    time.sleep(2)  # 2 seconds between runs

    # Save raw data
    df = pd.DataFrame(aggregated_results)
    df.to_csv(CSV_PATH, index=False)
    df.to_excel(EXCEL_PATH, index=False)

    # Save log
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))

    # Summary stats
    summary = df.groupby(["background", "volume", "gender", "keyword"]).agg({
        "wakeword_detected": ["mean", "std"],
        "tool_detected": ["mean", "std"]
    }).reset_index()

    summary.columns = ["background", "volume", "gender", "keyword",
                    "wakeword_accuracy", "wakeword_stdev",
                    "tool_accuracy", "tool_stdev"]

    # Generate bar plots
    for keyword in KEYWORDS:
        plt.figure()
        for gender in GENDERS:
            bars = []
            labels = []
            for bg in summary["background"].unique():
                row = summary[(summary["keyword"] == keyword) & (summary["gender"] == gender) & (summary["background"] == bg)]
                acc = row["tool_accuracy"].values[0] if not row.empty else 0
                bars.append(acc)
                labels.append(bg)
            plt.bar([f"{bg}\n({gender})" for bg in labels], bars, label=gender)
        plt.title(f"Tool Accuracy: {keyword}")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{keyword}_accuracy_comparison.png")
        plt.close()


        # Load CSV
    df = pd.read_csv(CSV_PATH)

    # Group by keyword, gender, background, and run, then count successful tool detections
    grouped = df.groupby(["keyword", "gender", "background", "run"])["tool_detected"].max().reset_index()
    detection_summary = grouped.groupby(["keyword", "gender", "background"])["tool_detected"].sum().reset_index()
    detection_summary["percent_detected"] = detection_summary["tool_detected"] * 10  # Convert 10/10 runs to 100%

    # Generate detection bar plots with error bars (std dev)
    for keyword in df["keyword"].unique():
    # Group by keyword, gender, background, and run, then summarize accuracy
        subset = df[df["keyword"] == keyword]
        grouped = subset.groupby(["keyword", "gender", "background"]).agg({
            "tool_detected": ["mean", "std"]
        }).reset_index()
        grouped.columns = ["keyword", "gender", "background", "mean_acc", "std_acc"]
        grouped["mean_acc"] *= 100  # Convert to percentage
        grouped["std_acc"] *= 100

        fig, ax = plt.subplots()
        labels = []
        bars = []
        errors = []

        for _, row in grouped.iterrows():
            label = f"{row['gender']} ({row['background']})"
            labels.append(label)
            bars.append(row["mean_acc"])
            errors.append(row["std_acc"])

        ax.bar(labels, bars, yerr=errors, capsize=5)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Detection Accuracy (%)")
        ax.set_title(f"Detection Accuracy after {N_RUNS} Runs: {keyword}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{keyword}_detection_rate_summary.png")
        plt.close()

        
    # Wakeword detection summary (bar plots with error bars)
    wakeword_summary = df.groupby(["gender", "background"]).agg({
        "wakeword_detected": ["mean", "std"]
    }).reset_index()
    wakeword_summary.columns = ["gender", "background", "mean_acc", "std_acc"]

    fig, ax = plt.subplots()
    labels = []
    bars = []
    errors = []

    for _, row in wakeword_summary.iterrows():
        label = f"{row['gender']} ({row['background']})"
        labels.append(label)
        bars.append(row["mean_acc"] * 100)
        errors.append(row["std_acc"] * 100)

    ax.bar(labels, bars, yerr=errors, capsize=5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Wakeword Detection Accuracy (%)")
    ax.set_title("Wakeword (ASTRA) Accuracy after 40 Runs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "wakeword_detection_summary.png")
    plt.close()


print("Script completed.")

