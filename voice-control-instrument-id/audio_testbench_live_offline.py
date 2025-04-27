#!/usr/bin/env python3
from audio_utils_offline import listen_and_transcribe_live

if __name__ == "__main__":
    tools = listen_and_transcribe_live()
    print("Final instruments:", tools)
