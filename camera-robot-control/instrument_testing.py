import os 
import sys
import cv2
import numpy as np
import math
import pyrealsense2 as rs
import time
import struct
import socket
import torch
# Import modules from voice-control-instrument-id/voice_instrument_functions.py
# instrument_module_path = os.path.abspath(os.path.join(__file__, "..", "..", "voice-control-instrument-id"))
# sys.path.append(instrument_module_path) 
# from voice_instrument_functions import *

# Load voice and instrument identification models
# porcupine, cobra, recorder = load_voice_model()
# inst_model = load_model('./voice-control-instrument-id/models/instrument_detector_model.pt', False)

from RayhansCoordinateTransformNew import * # Because of mediapipe, need to import AFTER loading instrument id model

# # Set up MyCoBot 280
HOST = "10.42.0.1"
GET_COORDS_PORT = 5006
MOVE_COORDS_PORT = 5005
MOVE_GRIPPER_PORT = 5007
home = [62.5, 81.8, 305.2, -177.21, -2.56, 45.91]

# # Set up Mediapipe hand tracking
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)

# Set up depth camera
# TODO: open as a window for Design Day
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

send_coords(home, HOST, MOVE_COORDS_PORT) # send robot to home
time.sleep(2) # TODO: better way to wait for arm to be stable before getting frame of tray

i = 40
img_name = './test-images/inst-camera'

while True:
    input("Press enter to get image")
    img_path = img_name + '_' + str(i) + '.png'
    frames = pipeline.poll_for_frames()
    # depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("Failed to get frames")
    else:
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(img_path, color_image)
    
    print(img_path)
    i += 1