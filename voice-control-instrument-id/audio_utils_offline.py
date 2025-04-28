#!/usr/bin/env python3
import os
import sys
import tempfile
import subprocess
import re

import speech_recognition as sr
import pyttsx3
import librosa
import soundfile as sf

# ─── CONFIG ────────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(__file__)
MODEL_FILE  = os.path.join(HERE, "whisper.cpp", "models", "ggml-small.en.bin")
CLI_BINARY  = os.path.join(HERE, "whisper.cpp", "build", "bin", "whisper-cli")

# all the ways someone might say “astra”
WAKE_WORDS = [
    "astra", "hey astra", "astraa", "austrah", "extra", "ast",
    "astra give", "hey astra give", "aster", "astro", "austro",
    "arstrah", "estra", "ausstra", "ausstrah", "ashtar", "asher",
    "astra,"
]

# synonyms for each instrument
ALT_NAMES = {
    "scissors": [
        "scissors","cesaurus","scizzards","sizzlers","sizzors","sizzers",
        "sizzars","scissor","seesaws","sizars","scizzles","cicero","cissors",
        "sizzled","seizer","cizzars"
    ],
    "forceps": [
        "forceps","four steps","for steps","4 steps","foursteps","forsteps","4steps",
        "four seps","for step","4 step","fourstep","forstep","4step","fourceps",
        "4ceps","four ceps","for ceps","4 ceps","foursips","forsips","4sips",
        "four sips","for sips","4 sips","foursip","forsip","4sip","four sip",
        "for sip","4 sip","foursep","forsep","4sep","four sep","for sep","4 sep",
        "forsets","for sets","4 sets","four sets","for sets","4 sets","forset",
        "for set","4 set","four set","for set","4 set","force","forces","forecepts",
        "forks ups","forsyths","forcepses","force apps","force epts","force eps",
        "forseps","for subs","four subs","forseize","fourseize","4 seize","4seps",
        "for-seps", "four-seps", "4-seps", "four-sep", "for-sep",
    ],
    "scalpel": [
        "scalpel","scalpels","scalpel blade","scalp","scale pill","scalball","skelple",
        "skull pill","scapple","scalp bell","scowl pill","sculpel","sculptor"
    ],
    "needle": [
        "needle","needles","kneadle","neato","knee doll","neadle","nidho","knead all",
        "noodle","kneel","neil","nido","nidole","nitto","meadle"
    ],
}

# phrases that put Astra to sleep
SLEEP_WORDS = [
    "astra go to sleep", "astra sleep", "go to sleep",
    "sleep", "bye astra", "goodbye astra"
]

# ─── SANITY CHECK ──────────────────────────────────────────────────────────────
for p in (MODEL_FILE, CLI_BINARY):
    if not os.path.isfile(p):
        sys.exit(f"❌ cannot find required file: {p}")

# ─── TTS SETUP ─────────────────────────────────────────────────────────────────
tts = pyttsx3.init()
for v in tts.getProperty("voices"):
    if "Rishi" in v.name:
        tts.setProperty("voice", v.id)
        break

def speak(text: str):
    print(f"[ASTRA] {text}")
    tts.say(text)
    tts.runAndWait()

# ─── MIC SETUP ─────────────────────────────────────────────────────────────────
recognizer = sr.Recognizer()
mic        = sr.Microphone()

# ─── run whisper-cli → plain text ───────────────────────────────────────────────
def transcribe_cli(wav: str) -> str:
    p = subprocess.run(
        [CLI_BINARY, "-m", MODEL_FILE, "-f", wav, "-otxt"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=True
    )
    return p.stdout.strip()

# ─── strip leading “[00:00:...] --> [...]” timestamps ───────────────────────────
def strip_timestamps(s: str) -> str:
    return re.sub(r"^\s*\[\d\d:\d\d:\d\d\.\d+\s*-->\s*\d\d:\d\d:\d\d\.\d+\]\s*", "", s)

# ─── map any alias back to its canonical instrument ────────────────────────────
def identify_instrument(cmd: str) -> str|None:
    lo = cmd.lower()
    for inst, alts in ALT_NAMES.items():
        for alias in ( [inst] + alts ):
            if alias in lo:
                return inst
    return None

# ─── TESTING ──────────────────────────────────────────────────────────────────
# ─── TESTING ──────────────────────────────────────────────────────────────────
def process_mixed_audio_with_background_and_wakeword(background_path, voice_files_info, background_offset_ms=0, background_volume_percent=100, play_during_transcription=True):
    try:
        # Prepare background audio if available
        background_wav = ensure_wav_format(background_path) if background_path else None
        temp_files = [background_wav] if background_wav else []

        if background_wav:
            full_bg = normalize_audio(AudioSegment.from_wav(background_wav)).set_frame_rate(16000).set_channels(1)
            background_audio = full_bg[background_offset_ms:]
            bg_volume_db = db_from_percentage(background_volume_percent)
            background_audio = background_audio + bg_volume_db
        else:
            background_audio = AudioSegment.silent(duration=15000)

        # Overlay voice files with background audio
        combined = background_audio
        for info in voice_files_info:
            voice_wav = ensure_wav_format(info["path"])
            if voice_wav != info["path"]:
                temp_files.append(voice_wav)
            voice_audio = normalize_audio(AudioSegment.from_wav(voice_wav)).set_frame_rate(16000).set_channels(1)
            voice_audio = voice_audio + db_from_percentage(info.get("volume", 100))
            combined = combined.overlay(voice_audio, position=info.get("start_ms", 0))

        # Create final mixed audio file
        max_end = max(info.get("start_ms", 0) + AudioSegment.from_wav(ensure_wav_format(info["path"])).duration_seconds * 1000 for info in voice_files_info)
        trim_point = int(max_end + 5000)
        combined = AudioSegment.silent(duration=1000) + combined[:trim_point] + AudioSegment.silent(duration=1000)

        final_mix_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        combined.export(final_mix_path, format="wav")

        if play_during_transcription:
            subprocess.run(["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", final_mix_path])

        # Use whisper-cli to transcribe the audio
        result = transcribe_cli(final_mix_path)  # Using the `transcribe_cli` function instead of whisper_model.transcribe

        transcription = result.strip().lower()

        wake_conf = 1.0 if any(w in transcription.lower().split() for w in WAKE_WORDS) else (
            0.7 if "astra" in transcription.lower() else 0.0
        )

        # Identify tool from transcription
        instruments = identify_instrument(transcription)
        if instruments:
            tool, conf = instruments[0]
            return transcription, final_mix_path, wake_conf, tool, conf

        return transcription, final_mix_path, wake_conf, None, 0.0

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None, 0.0, None, 0.0


# --- ensure wav format ───────────────────────────────────────────────────────────
from pydub import AudioSegment

def ensure_wav_format(input_path):
    input_path = str(input_path)
    if input_path.lower().endswith(".wav"):
        return input_path
    audio = AudioSegment.from_file(input_path)
    temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav_file.name, format="wav")
    return temp_wav_file.name

def normalize_audio(audio: AudioSegment):
    """
    Normalize the audio to have a consistent volume.
    """
    return audio.apply_gain(-audio.max_dBFS)  # Normalize to -1 dBFS

import numpy as np

def db_from_percentage(volume_percent):
    """
    Convert a volume percentage (0-100) to a decibel change (dB).
    0% volume -> -60 dB (completely silent)
    100% volume -> 0 dB (no change)
    """
    return np.interp(volume_percent, [0, 100], [-60, 0])

# ─── Fuzzy matching for instrument identification ────────────────────────────────

import difflib

def identify_instrument(cmd: str):
    instruments = ['scissors', 'scalpel', 'forceps', 'needle']
    alt_names = ALT_NAMES

    found = {}
    cleaned_cmd = cmd.lower()
    for instrument in instruments:
        if instrument in cleaned_cmd:
            found[instrument] = 1.0  # Assign a confidence value

    for inst, alts in alt_names.items():
        for alt in alts:
            if alt in cleaned_cmd:
                found[inst] = 0.9  # Assign a lower confidence for alt names

    for inst, conf in found.items():
        if conf >= 0.9:
            print(f"Instrument identified: {inst} (Confidence: {conf * 100:.0f}%)")
        else:
            print(f"Instrument identified (Fuzzy Match): {inst} (Confidence: ~{conf * 100:.0f}%)")
    return list(found.items())


# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
def listen_and_transcribe_live():
    speak("Astra ready.")
    print("⏺ Say “Astra” to begin…")

    while True:
        # 1) listen for up to 4 sec
        with mic as src:
            #recognizer.adjust_for_ambient_noise(src)
            print("🎤 Listening for wake word…")
            audio = recognizer.listen(src, phrase_time_limit=4)

        # 2) save & resample 16 kHz mono
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio.get_wav_data()); tmp.close()
        y, _ = librosa.load(tmp.name, sr=16000, mono=True)
        sf.write(tmp.name, y, 16000)

        
        
        # 3) whisper → strip timestamps, cleanup and filter out spurious digit-only “transcripts”
        raw = transcribe_cli(tmp.name)
        os.unlink(tmp.name)  # delete temp file immediately

        # strip leading timestamps and trim whitespace
        heard = strip_timestamps(raw).strip()

        # **NEW** remove any leftover numbers
        heard = re.sub(r'\d+', '', heard).strip()

        # ignore blank or pure-digit results
        if not heard:
            continue

        print(f"[HEARD] {heard}")  # echo what Astra actually heard
        lo = heard.lower()         # lowercase for wake-word/command logic


        # 4) sleep?
        if any(phr in lo for phr in SLEEP_WORDS):
            speak("Going to sleep.")
            print("💤 Exiting.")
            return

        # 5) wake-word?
        wake = next((w for w in WAKE_WORDS if w in lo), None)
        if not wake:
            continue

        # 6) see if user bundled a command after the wake phrase
        after = lo.replace(wake, "", 1)
        # strip out all punctuation & whitespace
        cmd_clean = re.sub(r"[^\w\s]", "", after).strip()
        if cmd_clean:
            inst = identify_instrument(cmd_clean)
            if inst:
                speak(f"Getting {inst}")
            else:
                speak(cmd_clean)
            continue

        # 7) else prompt for a second utterance
        speak("Listening.")
        with mic as src2:
            #recognizer.adjust_for_ambient_noise(src2)
            cmd_audio = recognizer.listen(src2, phrase_time_limit=4)

        tmp2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp2.write(cmd_audio.get_wav_data()); tmp2.close()
        y2, _ = librosa.load(tmp2.name, sr=16000, mono=True)
        sf.write(tmp2.name, y2, 16000)

         # 8) second utterance → strip timestamps, cleanup, filter out digit‐only
        raw2 = transcribe_cli(tmp2.name)
        os.unlink(tmp2.name)

        # strip leading “[00:…]-->[…]” blocks and trim
        cmd2 = strip_timestamps(raw2).strip()
        cmd2 = re.sub(r'\d+', '', cmd2).strip()

        # if it's empty or just digits/spaces, ignore and restart loop
        if not cmd2:
            continue

        print(f"[COMMAND] {cmd2}")
        inst2 = identify_instrument(cmd2)
        if inst2:
            speak(f"Getting {inst2}")
        else:
            speak(cmd2)


# ─── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    listen_and_transcribe_live()
