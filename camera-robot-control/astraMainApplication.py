# from RayhansCoordinateTransform import move_to_hand
import os 
import sys
import cv2
import numpy as np
import math
# import mediapipe as mp
import pyrealsense2 as rs
import time
import struct
import socket
import torch
instrument_module_path = os.path.abspath(os.path.join(__file__, "..", "..", "voice_control_instrument_id"))
sys.path.append(instrument_module_path) 
from voice_instrument_functions import *

# porcupine, cobra, recorder = load_voice_model()
# command = get_voice_command(Wporcupine, cobra, recorder)

# inst = get_instrument_name(command)

model = load_model('./voice_control_instrument_id/models/instrument_detector_model.pt', False)

#x_mid, y_midW = identify_instrument(model, os.path.abspath('./voice_control_instrument_id/eval-images/inst-notape.jpg'), inst)

# HOST = "10.42.0.1"
# GET_COORDS_PORT = 5006
# MOVE_COORDS_PORT = 5005
# home = [62.5, 81.8, 305.2, -177.21, -2.56, 45.91]
# move_to_hand()