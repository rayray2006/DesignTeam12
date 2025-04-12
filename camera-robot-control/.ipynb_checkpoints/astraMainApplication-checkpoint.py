from RayhansCoordinateTransform import move_to_hand
import cv2
import numpy as np
import math
import mediapipe as mp
import pyrealsense2 as rs
import time
import struct
import socket

HOST = "10.42.0.1"
GET_COORDS_PORT = 5006
MOVE_COORDS_PORT = 5005
home = [62.5, 81.8, 305.2, -177.21, -2.56, 45.91]
move_to_hand()