from RayhansCoordinateTransform import move_to_hand
from RayhansCoordinateTransform import send_coords
from RayhansCoordinateTransform import send_gripper_command
import cv2
import numpy as np
import math
import mediapipe as mp
import pyrealsense2 as rs
import time
import struct
import socket

HOST = "172.20.10.2"
GET_COORDS_PORT = 5006
MOVE_COORDS_PORT = 5005
MOVE_GRIPPER_PORT = 5007
home = [62.5, 81.8, 305.2, -177.21, -2.56, 45.91]
send_gripper_command(100, 100)
time.sleep(2)
send_gripper_command(0, 100)
time.sleep(2)
move_to_hand()