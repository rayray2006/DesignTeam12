import socket
import struct
import threading
import time
from pymycobot.mycobot import MyCobot

# Initialize MyCobot on the Pi.
mc = MyCobot('/dev/ttyAMA0', 1000000)

# Global variable to store the latest coordinates.
current_coords = None
coords_lock = threading.Lock()

def poll_coords():
    """Continuously poll the MyCobot for coordinates and update current_coords."""
    global current_coords
    while True:
        try:
            coords = mc.get_coords()
            with coords_lock:
                current_coords = coords
        except Exception as e:
            print("❌ Error polling coordinates:", e)
        time.sleep(0.1)  # Poll every 100ms.

# Define ports:
RECEIVE_PORT = 5005     # For receiving target coordinates (to move the robot)
RETURN_PORT = 5006      # For sending current coordinates upon request
GRIPPER_PORT = 5007     # For receiving gripper commands

def serve_target_coords():
    """
    Listens on RECEIVE_PORT for incoming 6-float binary payloads.
    When received, it commands the robot arm to move to the specified coordinates.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", RECEIVE_PORT))
    sock.listen(5)
    print(f"📡 Ready to receive target coordinates on port {RECEIVE_PORT}...")

    while True:
        conn, addr = sock.accept()
        try:
            data = conn.recv(1024)  # Expecting 24 bytes for 6 floats.
            if not data:
                conn.close()
                continue

            if len(data) == struct.calcsize("6fi"):
                unpacked = struct.unpack("6fi", data)

                # First 6 elements are the floats
                target_coords = list(unpacked[:6])

                # Last element is the integer
                type = unpacked[6]




                print("Received target coordinates:", target_coords)
                # Command the robot to move to the specified coordinates.
                if mc.is_moving() == 0:
                    if (type == 0):
                        mc.send_coords(target_coords, 100, 0)
                    else:
                        mc.send_angles(target_coords, 100)
            else:
                print("Received data of unexpected length:", len(data))
                conn.sendall(b"ERROR")
        except Exception as e:
            print(f"❌ Error handling target coordinates request: {e}")
        finally:
            conn.close()

def return_coords():
    """
    Listens on RETURN_PORT for a GET request.
    If a request starting with "GET_COORDS" is received, the current coordinates are sent.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", RETURN_PORT))
    sock.listen(5)
    print(f"📡 Ready to serve current coordinates on port {RETURN_PORT}...")

    while True:
        conn, addr = sock.accept()
        try:
            data = conn.recv(1024)  # Expecting a command like "GET_COORDS".
            if not data:
                conn.close()
                continue

            if data.startswith(b"GET_COORDS"):
                with coords_lock:
                    coords = current_coords
                if coords:
                    conn.sendall(struct.pack("6f", *coords))
                else:
                    print("⚠️ No current coordinates available.")
                    conn.sendall(b"ERROR")
            else:
                print("Received unknown command")
                conn.sendall(b"ERROR")
        except Exception as e:
            print(f"❌ Error handling coordinates request: {e}")
        finally:
            conn.close()

def serve_gripper():
    """
    Listens on GRIPPER_PORT for incoming 2-float binary payloads.
    When received, it commands the robot gripper to change state using set_gripper(state, speed).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", GRIPPER_PORT))
    sock.listen(5)
    print(f"📡 Ready to receive gripper commands on port {GRIPPER_PORT}...")

    while True:
        conn, addr = sock.accept()
        try:
            data = conn.recv(1024)  # Expecting 8 bytes for 2 floats.
            if not data:
                conn.close()
                continue

            if len(data) == struct.calcsize("2f"):
                state, speed = struct.unpack("2f", data)
                print("Received gripper command:", state, speed)
                mc.clear_error_information()
                mc.set_gripper_value(int(state), int(speed))
            else:
                print("Received data of unexpected length for gripper command:", len(data))
                conn.sendall(b"ERROR")
        except Exception as e:
            print(f"❌ Error handling gripper command request: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    # Start the dedicated polling thread.
    polling_thread = threading.Thread(target=poll_coords, daemon=True)
    polling_thread.start()

    # Start servers concurrently.
    t1 = threading.Thread(target=serve_target_coords, daemon=True)
    t2 = threading.Thread(target=return_coords, daemon=True)
    t3 = threading.Thread(target=serve_gripper, daemon=True)
    t1.start()
    t2.start()
    t3.start()

    print("Servers are running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.25)  # Allow CPU time for other threads.
    except KeyboardInterrupt:
        print("Exiting.")
