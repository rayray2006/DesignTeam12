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
    "arstrah", "estra", "ausstra", "ausstrah", "ashtar", "asher"
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
        "forseps","for subs","four subs","forseize","fourseize","4 seize","4seps"
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
