import socket
import struct
import threading
import time
from pymycobot.mycobot import MyCobot

# Initialize MyCobot on the Pi.
mc = MyCobot('/dev/ttyAMA0', 1000000)
mc.release_servo(4)
time.sleep(5)
mc.set_servo_calibration(4)
time.sleep(1)
mc.send_angle(4, 0, 10)