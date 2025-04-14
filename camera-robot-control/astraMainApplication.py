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
instrument_module_path = os.path.abspath(os.path.join(__file__, "..", "..", "voice-control-instrument-id"))
sys.path.append(instrument_module_path) 
from voice_instrument_functions import *

# Load voice and instrument identification models
porcupine, cobra, recorder = load_voice_model()
inst_model = load_model('./voice-control-instrument-id/models/instrument_detector_model.pt', False)

from RayhansCoordinateTransformNew import * # Because of mediapipe, need to import AFTER loading instrument id model

# Set up MyCoBot 280
HOST = "172.20.10.2"
GET_COORDS_PORT = 5006
MOVE_COORDS_PORT = 5005
MOVE_GRIPPER_PORT = 5007
home = [62.5, 81.8, 305.2, -177.21, -2.56, 45.91]

# Set up Mediapipe hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Set up depth camera
# TODO: open as a window for Design Day
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

while True:
    send_coords(home, HOST, MOVE_COORDS_PORT) # send robot to home
    time.sleep(2) # TODO: better way to wait for arm to be stable before getting frame of tray
    inst_img = get_camera_img(pipeline) # get png of tray to run instrument id

    command = get_voice_command(porcupine, cobra, recorder) # get voice command
    inst = get_instrument_name(command) # transcribe voice command and get name of instrument
    x_mid, y_mid = identify_instrument(inst_model, inst_img, inst) # get 2d coords of instrument

    ## Get color and depth frame
    frames = pipeline.poll_for_frames()
    if not frames:
        continue

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert yolo instrument coords to 3D coords
    inst_coords = get_inst_coords(color_frame, depth_frame, x_mid, y_mid)

    target_coords = inst_coords + [home[3], home[4], home[5]]
    send_coords(target_coords)

    # Pseudo code of what's next:
        # send coords to instrument. For base_coords (0,1,2) use inst_coords. For (3,4,5) use whatever the euler angles are to make the end effector in line with the instrument
        # pick up instrument
        # go into position to look for hand (same as home?)
        # go to hand
        # release instrument
        # go back home/looking at tray
