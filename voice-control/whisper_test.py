import tkinter as tk
import logging
import whisper
import pyaudio
import wave
import os

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
WAKE_WORD = "alexa"

# Load Whisper model (this may take a few seconds)
logging.basicConfig(level=logging.INFO)
logging.info("Loading Whisper model...")
model = whisper.load_model("base")
logging.info("Whisper model loaded.")

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Global variables for audio buffers and state
audio_buffer = []      # Buffer for continuously collected audio (wake word detection)
command_buffer = []    # Buffer for command recording after wake word
recording_command = False
listening = False
stream = None

# Custom logging handler to output logs to the Tkinter text widget
class TkinterHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)
        self.text_widget.after(0, append)

# Function to delete a temporary audio file
def delete_audio_file(filename):
    try:
        os.remove(filename)
        logging.info(f"Deleted temporary file: {filename}")
    except Exception as e:
        logging.error(f"Error deleting file {filename}: {e}")

# Function to transcribe an audio file using Whisper
def transcribe_audio(filename):
    logging.info(f"Transcribing audio from {filename}...")
    result = model.transcribe(filename)
    transcription = result["text"].lower()
    logging.info(f"Transcription result: {transcription}")
    return transcription

# PyAudio callback: continuously receives audio chunks.
# If we’re in command-recording mode, store chunks in command_buffer,
# otherwise accumulate in audio_buffer for wake word detection.
def audio_callback(in_data, frame_count, time_info, status):
    global audio_buffer, command_buffer, recording_command
    if recording_command:
        command_buffer.append(in_data)
    else:
        audio_buffer.append(in_data)
    return (None, pyaudio.paContinue)

# Function that runs every second to check the accumulated audio for the wake word.
def check_for_wake_word():
    global audio_buffer, listening, recording_command
    if not listening:
        return  # Stop if listening has been toggled off
    if audio_buffer:
        wake_file = "temp_wake.wav"
        try:
            # Save the current buffer to a temporary file for processing
            wf = wave.open(wake_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(audio_buffer))
            wf.close()
            logging.info("Checking for wake word in recorded audio...")
            transcription = transcribe_audio(wake_file)
            logging.info(f"Wake word check transcription: {transcription}")
            if WAKE_WORD in transcription:
                logging.info(f"Wake word '{WAKE_WORD}' detected!")
                start_command_recording()
            # Clear the wake word detection buffer after each check
            audio_buffer.clear()
        except Exception as e:
            logging.error(f"Error during wake word check: {e}")
        finally:
            delete_audio_file(wake_file)
    if listening:
        # Schedule the next check in 1000 ms
        root.after(1000, check_for_wake_word)

# Function to start a 5-second command recording once the wake word is detected.
def start_command_recording():
    global recording_command, command_buffer
    recording_command = True
    command_buffer.clear()  # Start with an empty command buffer
    logging.info("Started command recording for 5 seconds...")
    # After 5 seconds, process the command recording.
    root.after(5000, finish_command_recording)

# Finish the command recording, transcribe the recorded command, and log the results.
def finish_command_recording():
    global recording_command, command_buffer
    command_file = "temp_command.wav"
    try:
        wf = wave.open(command_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(command_buffer))
        wf.close()
        logging.info("Finished command recording. Transcribing command audio...")
        command_transcription = transcribe_audio(command_file)
        logging.info(f"Command transcription: {command_transcription}")
    except Exception as e:
        logging.error(f"Error during command transcription: {e}")
    finally:
        delete_audio_file(command_file)
        recording_command = False

# Toggle function to start/stop listening.
def toggle_listening():
    global listening, stream, audio_buffer, command_buffer, recording_command
    if not listening:
        listening = True
        toggle_button.config(text="Stop Listening")
        logging.info("Listening started.")
        audio_buffer.clear()
        command_buffer.clear()
        recording_command = False
        # Open and start the stream in non-blocking callback mode.
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK,
                            stream_callback=audio_callback)
        stream.start_stream()
        # Begin checking for the wake word every second.
        root.after(1000, check_for_wake_word)
    else:
        listening = False
        toggle_button.config(text="Start Listening")
        logging.info("Listening stopped.")
        if stream is not None:
            if stream.is_active():
                stream.stop_stream()
            stream.close()

# Set up the Tkinter UI
root = tk.Tk()
root.title("Dynamic Audio Listener")

# Create a text widget for logging
log_text = tk.Text(root, state='disabled', wrap='word', height=20, width=80)
log_text.pack(padx=10, pady=10)

# Create a toggle button for starting/stopping listening
toggle_button = tk.Button(root, text="Start Listening", command=toggle_listening)
toggle_button.pack(pady=10)

# Set up logging to the terminal and the Tkinter text widget
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

terminal_handler = logging.StreamHandler()
terminal_handler.setFormatter(formatter)
logger.addHandler(terminal_handler)

tk_handler = TkinterHandler(log_text)
tk_handler.setFormatter(formatter)
logger.addHandler(tk_handler)

# Start the Tkinter main event loop
root.mainloop()
