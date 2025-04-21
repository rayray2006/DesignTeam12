import socket
import struct
import threading
import time
from pymycobot.mycobot import MyCobot

# Initialize MyCobot on the Pi.
mc = MyCobot('/dev/ttyAMA0', 1000000)
mc.send_angles([0,0,0,0,0,0], 1)