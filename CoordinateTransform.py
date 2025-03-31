import cv2
import numpy as np
import math
import mediapipe as mp
import pyrealsense2 as rs
import time
import struct
import socket



# --- Raspberry Pi Connection Details ---
HOST = "10.42.0.1"
GET_COORDS_PORT = 5006
MOVE_COORDS_PORT = 5005
home = [-63, -79.1, 305.3, -177.03, 2.5, 135.14]


def get_coords():
    """Connects to the robot's coordinates server to retrieve current coordinates."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST, GET_COORDS_PORT))
            # Send the command to get current coordinates.
            sock.sendall(b"GET_COORDS")
            data = sock.recv(1024)
            
            # Check if the server returned an error.
            if data.startswith(b"ERROR"):
                print("Error: Failed to retrieve coordinates.")
                return None
            
            # Unpack the 6 float values (each float is 4 bytes, so expect 24 bytes total).
            if len(data) == struct.calcsize("6f"):
                coords = struct.unpack("6f", data)
                return coords
            else:
                print(f"Unexpected data length: {len(data)} bytes.")
                return None
    except Exception as e:
        print("An exception occurred:", e)
        return None

def send_coords(coords):
    """
    Connects to the robot's target coordinates server and sends a 6-float binary payload.
    
    Parameters:
        coords (list or tuple): A list or tuple containing 6 float values representing the target coordinates.
    """
    # Pack the coordinates into binary data (6 floats).
    data = struct.pack("6f", *coords)
    
    # Create a socket and connect to the robot's server.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.connect((HOST, MOVE_COORDS_PORT))
            sock.sendall(data)
            
            # Wait for and print the response from the server.
            response = sock.recv(1024)
            print("Response from robot:", response.decode())
        except Exception as e:
            print("Error connecting or sending data:", e)

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Create a rotation matrix from Euler angles (roll, pitch, yaw).
    Angles are in radians. Using intrinsic rotations in ZYX order.
    """
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
    
    # Combined rotation: R = R_z * R_y * R_x
    return R_z @ R_y @ R_x

def transform_camera_to_robot(camera_coords, end_effector_coords, euler_angles, angles_in_degrees=True):
    x_c, y_c, z_c = camera_coords
    X_ee, Y_ee, Z_ee = end_effector_coords
    roll, pitch, yaw = euler_angles

    if angles_in_degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
    
    R_ee = euler_to_rotation_matrix(roll, pitch, yaw)
    
    # Fixed permutation: new_x = camera_y, new_y = camera_z, new_z = camera_x.
    R_fixed = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    



    x_offset = 0  # replace with your desired offset in mm
    y_offset = 0
    z_offset = 0

    camera_vec = np.array([[x_c + x_offset], [y_c + y_offset], [z_c + z_offset]])



    
    transformed_change = R_ee @ (R_fixed @ camera_vec)
    
    # Multiply y and z changes by -1 before adding translation.
    
    robot_vec = np.array([[X_ee], [Y_ee], [Z_ee]]) + transformed_change
    
    return robot_vec.flatten()

def get_hand_coords(color_frame, depth_frame):
    color_image = np.asanyarray(color_frame.get_data())
    image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    # Process image for hand landmarks using the global 'hands' object
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            try:
                h, w, _ = color_image.shape
                
                # Use the index finger tip (landmark 9) as an example
                index_tip = hand_landmarks.landmark[9]
                pixel_x, pixel_y = int(index_tip.x * w), int(index_tip.y * h)
                # Ensure pixel coordinates are within image bounds
                pixel_x = max(0, min(pixel_x, w - 1))
                pixel_y = max(0, min(pixel_y, h - 1))
                
                # Get the depth at the pixel (in meters)
                depth_value = depth_frame.get_distance(pixel_x, pixel_y)
                if depth_value == 0:
                    continue
                
                # Get intrinsics from the color stream
                color_intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
                
                # Deproject the 2D pixel (with depth) to a 3D point (in meters)
                point_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics,
                                                           [pixel_x, pixel_y],
                                                           depth_value)
                # Convert from meters to millimeters
                point_3d_mm = [coord * 1000 for coord in point_3d]
                
                return point_3d_mm, pixel_x, pixel_y, color_image
            except Exception as e:
                print("Error processing hand landmarks:", e)
                continue
    return None, None, None, color_image

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Configure and start the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

try:
    none_counter = 0  # tracks consecutive frames without hand detection
    prev_coords = home
    while True:
        frames = pipeline.poll_for_frames()
        if not frames:
            continue

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        point_3d_mm, pixel_x, pixel_y, color_image = get_hand_coords(color_frame, depth_frame)

        if point_3d_mm is None:
            none_counter += 1
            if none_counter >= 10:
                print("No hand detected for 10 frames. Sending robot home.")
                send_coords(home)
                none_counter = 0  # reset after sending home
            continue
        else:
            none_counter = 0  # reset the counter on successful detection

        endEffectorCoords = get_coords()
        if endEffectorCoords is None:
            print("Robot did not return coordinates")
            continue

        end_effector = endEffectorCoords[:3]
        euler_angles = home  [3:]

        base_coords = transform_camera_to_robot(point_3d_mm, end_effector, euler_angles, angles_in_degrees=True)
        target_coords = np.concatenate((base_coords, euler_angles))
        if np.linalg.norm(np.array(target_coords) - np.array(prev_coords)) > 50:  # mm threshold
            send_coords(target_coords)
            prev_coords = target_coords
            
        time.sleep(1)


finally:
    pipeline.stop()
    cv2.destroyAllWindows()
