import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import math
import mediapipe as mp
import pyrealsense2 as rs
import socket
import struct
import time
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Global connection and home configuration ---
HOST = "10.42.0.1"
GET_COORDS_PORT = 5006
MOVE_COORDS_PORT = 5005
home = [-63, -79.1, 305.3, -177.03, 2.5, 135.14]

# --- Functions for robot communication ---
def get_coords():
    """Retrieve current robot coordinates with a short socket timeout."""
    try:
        logging.debug("Attempting to get robot coordinates...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)  # Short timeout for quick failure if not connected
            sock.connect((HOST, GET_COORDS_PORT))
            sock.sendall(b"GET_COORDS")
            data = sock.recv(1024)
            if data.startswith(b"ERROR"):
                logging.error("Robot returned ERROR in get_coords")
                return None
            if len(data) == struct.calcsize("6f"):
                coords = struct.unpack("6f", data)
                logging.debug(f"Received robot coords: {coords}")
                return coords
            else:
                logging.error(f"Unexpected data length in get_coords: {len(data)} bytes")
                return None
    except socket.timeout:
        logging.debug("get_coords timed out (robot likely not connected)")
        return None
    except Exception as e:
        logging.exception("Exception in get_coords:")
        return None

def send_coords(coords):
    """Send target coordinates with a short socket timeout."""
    try:
        logging.debug(f"Sending robot coordinates: {coords}")
        data = struct.pack("6f", *coords)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)  # Short timeout
            sock.connect((HOST, MOVE_COORDS_PORT))
            sock.sendall(data)
            response = sock.recv(1024)
            logging.info(f"Response from robot: {response.decode()}")
    except socket.timeout:
        logging.debug("send_coords timed out (robot likely not connected)")
    except Exception as e:
        logging.exception("Error connecting or sending data in send_coords:")

# --- Functions for coordinate transformation ---
def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    R_y = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    R_z = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    return R_z @ R_y @ R_x

def transform_camera_to_robot(camera_coords, end_effector_coords, euler_angles, angles_in_degrees=True):
    try:
        logging.debug(f"Transforming camera coords: {camera_coords} with end effector: {end_effector_coords} and Euler angles: {euler_angles}")
        x_c, y_c, z_c = camera_coords
        X_ee, Y_ee, Z_ee = end_effector_coords
        roll, pitch, yaw = euler_angles
        if angles_in_degrees:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)
        R_ee = euler_to_rotation_matrix(roll, pitch, yaw)
        # Identity transformation; adjust if needed.
        R_fixed = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
        camera_vec = np.array([[x_c], [y_c], [z_c]])
        transformed_change = R_ee @ (R_fixed @ camera_vec)
        robot_vec = np.array([[X_ee], [Y_ee], [Z_ee]]) + transformed_change
        transformed_coords = robot_vec.flatten()
        logging.debug(f"Transformed robot coords: {transformed_coords}")
        return transformed_coords
    except Exception as e:
        logging.exception("Error in transform_camera_to_robot:")
        return None

# --- MediaPipe hand detection initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_hand_coords(color_frame, depth_frame):
    """
    Processes the color frame to detect hand landmarks.
    Returns 3D hand coordinates (mm), pixel coordinates, and the raw color image.
    """
    try:
        color_image = np.asanyarray(color_frame.get_data())
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    h, w, _ = color_image.shape
                    index_tip = hand_landmarks.landmark[9]
                    pixel_x, pixel_y = int(index_tip.x * w), int(index_tip.y * h)
                    pixel_x = max(0, min(pixel_x, w - 1))
                    pixel_y = max(0, min(pixel_y, h - 1))
                    depth_value = depth_frame.get_distance(pixel_x, pixel_y)
                    if depth_value == 0:
                        continue
                    color_intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
                    point_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [pixel_x, pixel_y], depth_value)
                    point_3d_mm = [coord * 1000 for coord in point_3d]
                    logging.debug(f"Detected hand at pixel ({pixel_x}, {pixel_y}) with 3D coords: {point_3d_mm}")
                    return point_3d_mm, pixel_x, pixel_y, color_image
                except Exception as e:
                    logging.exception("Error processing hand landmarks:")
                    continue
        return None, None, None, color_image
    except Exception as e:
        logging.exception("Error in get_hand_coords:")
        return None, None, None, None

# --- Tkinter UI class ---
class RobotControlUI:
    def __init__(self, master):
        self.master = master
        self.master.title("MyCobot 280 Control Interface")
        self.latest_frame = None
        self.hand_coords = None      # Latest hand 3D coordinates (mm)
        self.target_coords = None    # Computed target coordinates for robot command
        self.robot_coords = None     # Latest robot coordinates (from get_coords())
        self.hand_detected = False
        self.command_queue = []
        self.camera_paused = False
        self.fps = 30
        self.lock = threading.Lock()
        self.running = True
        self.pipeline = None   # RealSense pipeline handle
        self.setup_ui()

    def setup_ui(self):
        logging.debug("Setting up UI...")
        left_frame = tk.Frame(self.master)
        left_frame.grid(row=0, column=0, padx=10, pady=10)
        right_frame = tk.Frame(self.master)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        self.video_label = tk.Label(left_frame)
        self.video_label.pack()
        self.hand_indicator = tk.Label(right_frame, text="Hand Detected: NO", font=("Arial", 12))
        self.hand_indicator.pack(pady=5)
        self.hand_coords_label = tk.Label(right_frame, text="Hand Coords: Not Detected", font=("Arial", 10))
        self.hand_coords_label.pack(pady=5)
        self.robot_coords_label = tk.Label(right_frame, text="Robot Coords: Unknown", font=("Arial", 10))
        self.robot_coords_label.pack(pady=5)
        self.target_coords_label = tk.Label(right_frame, text="Target Coords: Not Available", font=("Arial", 10))
        self.target_coords_label.pack(pady=5)
        fps_frame = tk.Frame(right_frame)
        fps_frame.pack(pady=5)
        tk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_slider = tk.Scale(fps_frame, from_=1, to=60, orient=tk.HORIZONTAL, length=150)
        self.fps_slider.set(30)
        self.fps_slider.pack(side=tk.LEFT)
        self.fps_slider.bind("<Motion>", lambda event: self.update_fps())
        buttons_frame = tk.Frame(right_frame)
        buttons_frame.pack(pady=10)
        home_button = tk.Button(buttons_frame, text="Home", command=self.send_home)
        home_button.grid(row=0, column=0, padx=5, pady=5)
        send_button = tk.Button(buttons_frame, text="Send Command", command=self.send_command)
        send_button.grid(row=0, column=1, padx=5, pady=5)
        pause_button = tk.Button(buttons_frame, text="Pause Camera", command=self.pause_camera)
        pause_button.grid(row=1, column=0, padx=5, pady=5)
        resume_button = tk.Button(buttons_frame, text="Resume Camera", command=self.resume_camera)
        resume_button.grid(row=1, column=1, padx=5, pady=5)
        clear_button = tk.Button(buttons_frame, text="Clear Queue", command=self.clear_command_queue)
        clear_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        queue_frame = tk.Frame(right_frame)
        queue_frame.pack(pady=10)
        tk.Label(queue_frame, text="Command Queue:").pack()
        self.queue_listbox = tk.Listbox(queue_frame, width=40, height=10)
        self.queue_listbox.pack(side=tk.LEFT, fill=tk.BOTH)
        scrollbar = tk.Scrollbar(queue_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.queue_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.queue_listbox.yview)
        logging.debug("UI setup complete.")

    def start(self):
        logging.info("Starting application...")
        self.initialize_pipeline()
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        self.robot_coords_thread = threading.Thread(target=self.update_robot_coords_loop, daemon=True)
        self.robot_coords_thread.start()
        self.update_camera_feed()

    def initialize_pipeline(self):
        """Initialize the RealSense pipeline."""
        try:
            if self.pipeline is not None:
                logging.debug("Stopping existing pipeline...")
                self.pipeline.stop()
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            logging.info("RealSense pipeline initialized.")
        except Exception as e:
            logging.exception("Failed to initialize pipeline:")
            self.pipeline = None

    def camera_loop(self):
        none_counter = 0
        local_home = home
        while self.running:
            try:
                if self.camera_paused:
                    time.sleep(0.1)
                    continue

                if self.pipeline is None:
                    logging.debug("Pipeline not initialized. Attempting reinitialization...")
                    self.initialize_pipeline()
                    time.sleep(1)
                    continue

                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                except Exception as fe:
                    logging.error("Frame timeout or error: %s", fe)
                    time.sleep(0.1)
                    continue

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    logging.debug("Depth or color frame not available.")
                    time.sleep(0.1)
                    continue

                hand_point, pixel_x, pixel_y, color_image = get_hand_coords(color_frame, depth_frame)
                with self.lock:
                    self.latest_frame = color_image.copy()
                    if hand_point is None:
                        self.hand_detected = False
                        none_counter += 1
                        self.hand_coords = None
                        self.target_coords = None
                        if none_counter >= 10:
                            try:
                                send_coords(local_home)
                                self.add_command("Home command sent (no hand detected for 10 frames)")
                            except Exception as e:
                                logging.exception("Error sending home command:")
                            none_counter = 0
                    else:
                        none_counter = 0
                        self.hand_detected = True
                        self.hand_coords = hand_point
                        current_robot = get_coords()
                        if current_robot is not None:
                            end_effector = current_robot[:3]
                            euler_angles = local_home[3:]
                            base_coords = transform_camera_to_robot(hand_point, end_effector, euler_angles, angles_in_degrees=True)
                            if base_coords is not None:
                                self.target_coords = np.concatenate((base_coords, euler_angles))
                            else:
                                self.target_coords = None
                        else:
                            self.target_coords = None
            except Exception as loop_e:
                logging.exception("Exception in camera loop iteration:")
            time.sleep(1/30)

    def update_robot_coords_loop(self):
        while self.running:
            try:
                coords = get_coords()
                with self.lock:
                    self.robot_coords = coords
            except Exception as e:
                logging.exception("Exception in update_robot_coords_loop:")
            time.sleep(0.5)

    def update_camera_feed(self):
        with self.lock:
            frame = self.latest_frame
            hand_detected = self.hand_detected
            hand_coords = self.hand_coords
            robot_coords = self.robot_coords
            target_coords = self.target_coords

        if frame is not None:
            try:
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv2image)
            except Exception as e:
                logging.exception("Error converting frame:")
                pil_image = Image.new("RGB", (640, 480), (0, 0, 0))
                draw = ImageDraw.Draw(pil_image)
                draw.text((10, 10), "Frame conversion error", fill="red")
        else:
            pil_image = Image.new("RGB", (640, 480), (50, 50, 50))
            draw = ImageDraw.Draw(pil_image)
            draw.text((10, 10), "No Camera Feed", fill="white")

        imgtk = ImageTk.PhotoImage(image=pil_image)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        if hand_coords is not None:
            hand_text = f"Hand Coords (mm): X={hand_coords[0]:.1f}, Y={hand_coords[1]:.1f}, Z={hand_coords[2]:.1f}"
        else:
            hand_text = "Hand Coords: Not Detected"
        self.hand_coords_label.config(text=hand_text)

        if hand_detected:
            self.hand_indicator.config(text="Hand Detected: YES", fg="green")
        else:
            self.hand_indicator.config(text="Hand Detected: NO", fg="red")

        if robot_coords is not None:
            robot_text = "Robot Coords: " + ", ".join(f"{c:.1f}" for c in robot_coords)
        else:
            robot_text = "Robot Coords: Unknown"
        self.robot_coords_label.config(text=robot_text)

        if target_coords is not None:
            target_text = "Target Coords: " + ", ".join(f"{c:.1f}" for c in target_coords)
        else:
            target_text = "Target Coords: Not Available"
        self.target_coords_label.config(text=target_text)

        fps_value = self.fps_slider.get()
        delay = int(1000 / fps_value) if fps_value > 0 else 33
        self.master.after(delay, self.update_camera_feed)

    def add_command(self, command_str):
        with self.lock:
            self.command_queue.append(command_str)
        self.queue_listbox.insert(tk.END, command_str)
        self.queue_listbox.yview_moveto(1)
        logging.debug(f"Command added: {command_str}")

    def clear_command_queue(self):
        with self.lock:
            self.command_queue = []
        self.queue_listbox.delete(0, tk.END)
        logging.debug("Command queue cleared.")

    def send_command(self):
        with self.lock:
            if self.target_coords is not None:
                coords_to_send = self.target_coords
            else:
                messagebox.showwarning("Warning", "No target coordinates available!")
                return
        try:
            send_coords(coords_to_send)
            self.add_command("Sent command: " + ", ".join(f"{c:.1f}" for c in coords_to_send))
        except Exception as e:
            logging.exception("Exception sending command:")

    def send_home(self):
        try:
            send_coords(home)
            self.add_command("Sent Home command")
        except Exception as e:
            logging.exception("Exception sending home:")

    def pause_camera(self):
        self.camera_paused = True
        self.add_command("Camera feed paused")

    def resume_camera(self):
        self.camera_paused = False
        self.add_command("Camera feed resumed")

    def update_fps(self):
        self.fps = self.fps_slider.get()

    def on_closing(self):
        logging.info("Closing application...")
        self.running = False
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                logging.exception("Exception stopping pipeline:")
        self.master.destroy()

def main():
    root = tk.Tk()
    app = RobotControlUI(root)
    app.start()
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
