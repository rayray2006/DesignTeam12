import socket
import struct
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from pymycobot.mycobot import MyCobot

# Configure logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Initialize MyCobot on the Pi.
mc = MyCobot('/dev/ttyAMA0', 1000000)

# Global variable to store the latest coordinates.
current_coords = None
coords_lock = threading.Lock()

def recv_all(conn, length):
    """
    Helper function to receive exactly `length` bytes from the connection.
    Raises a ConnectionError if the socket closes before all data is received.
    """
    data = b""
    while len(data) < length:
        more = conn.recv(length - len(data))
        if not more:
            raise ConnectionError("Socket closed before receiving full data")
        data += more
    return data

def poll_coords():
    """Continuously poll the MyCobot for coordinates and update current_coords."""
    global current_coords
    while True:
        try:
            coords = mc.get_coords()
            with coords_lock:
                current_coords = coords
        except Exception as e:
            logging.error("Error polling coordinates: %s", e)
        time.sleep(0.1)  # Poll every 100ms.

# Define ports.
RECEIVE_PORT = 5005     # For receiving target coordinates (to move the robot)
RETURN_PORT = 5006      # For sending current coordinates upon request

def handle_target_coords(conn, addr):
    """Handles a single connection for receiving target coordinates."""
    with conn:
        try:
            expected_bytes = struct.calcsize("6f")  # 24 bytes expected for 6 floats.
            data = recv_all(conn, expected_bytes)
            if len(data) == expected_bytes:
                target_coords = struct.unpack("6f", data)
                logging.info("Received target coordinates from %s: %s", addr, target_coords)
                # Command the robot to move to the specified coordinates.
                mc.clear_error_information()
                mc.send_coords(target_coords, 100, 0)
            else:
                logging.warning("Received data of unexpected length from %s: %d", addr, len(data))
                conn.sendall(b"ERROR")
        except Exception as e:
            logging.error("Error handling target coordinates request from %s: %s", addr, e)
            try:
                conn.sendall(b"ERROR")
            except Exception:
                pass

def serve_target_coords():
    """
    Listens on RECEIVE_PORT for incoming 6-float binary payloads.
    When received, it commands the robot arm to move to the specified coordinates.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", RECEIVE_PORT))
        sock.listen(5)
        logging.info("Ready to receive target coordinates on port %d...", RECEIVE_PORT)

        # Use a ThreadPoolExecutor to handle multiple connections concurrently.
        with ThreadPoolExecutor(max_workers=10) as executor:
            while True:
                try:
                    conn, addr = sock.accept()
                    executor.submit(handle_target_coords, conn, addr)
                except Exception as e:
                    logging.error("Error accepting connection on target coords server: %s", e)

def handle_return_coords(conn, addr):
    """Handles a single connection for returning the current coordinates."""
    with conn:
        try:
            data = conn.recv(1024)  # Expecting a command like "GET_COORDS".
            if not data:
                return

            if data.startswith(b"GET_COORDS"):
                with coords_lock:
                    coords = current_coords
                if coords:
                    conn.sendall(struct.pack("6f", *coords))
                else:
                    logging.warning("No current coordinates available for %s.", addr)
                    conn.sendall(b"ERROR")
            else:
                logging.warning("Received unknown command from %s", addr)
                conn.sendall(b"ERROR")
        except Exception as e:
            logging.error("Error handling coordinates request from %s: %s", addr, e)
            try:
                conn.sendall(b"ERROR")
            except Exception:
                pass

def return_coords():
    """
    Listens on RETURN_PORT for a GET request.
    If a request starting with "GET_COORDS" is received, the current coordinates are sent.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", RETURN_PORT))
        sock.listen(5)
        logging.info("Ready to serve current coordinates on port %d...", RETURN_PORT)

        with ThreadPoolExecutor(max_workers=10) as executor:
            while True:
                try:
                    conn, addr = sock.accept()
                    executor.submit(handle_return_coords, conn, addr)
                except Exception as e:
                    logging.error("Error accepting connection on return coords server: %s", e)

if __name__ == "__main__":
    # Start the dedicated polling thread.
    polling_thread = threading.Thread(target=poll_coords, daemon=True)
    polling_thread.start()

    # Start both servers concurrently.
    t1 = threading.Thread(target=serve_target_coords, daemon=True)
    t2 = threading.Thread(target=return_coords, daemon=True)
    t1.start()
    t2.start()

    logging.info("Servers are running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.25)  # Allow CPU time for other threads.
    except KeyboardInterrupt:
        logging.info("Exiting.")
