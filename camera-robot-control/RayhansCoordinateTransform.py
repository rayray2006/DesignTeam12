# We need to fix the sensing of the hand in the place_tool function

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
MOVE_GRIPPER_PORT = 5007
home = [62.5, 81.8, 305.2, -177.21, -2.56, 45.91]
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
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



def send_gripper_command(state, speed):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST, MOVE_GRIPPER_PORT))
            data = struct.pack("2f", state, speed)
            sock.sendall(data)
            print("Gripper command sent:", state, speed)
    except Exception as e:
        print("Error sending gripper command:", e)



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
    



    x_offset = 75  # replace with your desired offset in mm
    y_offset = -35
    z_offset = -100

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
                index_tip = hand_landmarks.landmark[5]
                wrist = hand_landmarks.landmark[0]
                otherFinger = hand_landmarks.landmark[17]
                indexpixel_x, indexpixel_y = int(index_tip.x * w), int(index_tip.y * h)
                wristpixel_x, wristpixel_y = int(wrist.x * w), int(wrist.y * h)
                otherpixel_x, otherpixel_y = int(otherFinger.x * w), int(otherFinger.y * h)
                # Ensure pixel coordinates are within image bounds
                indexpixel_x = max(0, min(indexpixel_x, w - 1))
                indexpixel_y = max(0, min(indexpixel_y, h - 1))
                wristpixel_x = max(0, min(wristpixel_x, w - 1))
                wristpixel_y = max(0, min(wristpixel_y, h - 1))
                otherpixel_x = max(0, min(otherpixel_x, w - 1))
                otherpixel_y = max(0, min(otherpixel_y, h - 1))
                
                # Get the depth at the pixel (in meters)
                indexdepth_value = depth_frame.get_distance(indexpixel_x, indexpixel_y)
                otherdepth_value = depth_frame.get_distance(otherpixel_x, otherpixel_y)
                wristdepth_value = depth_frame.get_distance(wristpixel_x, wristpixel_y)
                if indexdepth_value == 0 or wristdepth_value == 0 or otherdepth_value==0:
                    continue
                wristdepth_value = otherdepth_value
                
                
                # Get intrinsics from the color stream
                color_intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
                
                # Deproject the 2D pixel (with depth) to a 3D point (in meters)
                indexpoint_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics,
                                                           [indexpixel_x, indexpixel_y],
                                                           indexdepth_value)
                wristpoint_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics,
                                                           [wristpixel_x, wristpixel_y],
                                                           wristdepth_value)
                # Convert from meters to millimeters
                indexpoint_3d_mm = [coord * 1000 for coord in indexpoint_3d]
                wristpoint_3d_mm = [coord * 1000 for coord in wristpoint_3d]

                theta = math.radians(44)

                # Rotation matrix for a rotation around the z-axis:
                R_z = np.array([
                    [math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta),  math.cos(theta), 0],
                    [0,                0,               1]
                ])

                # Example coordinate vector
                indexcoord = np.array(indexpoint_3d_mm)
                wristcoord = np.array(wristpoint_3d_mm)

                # Transform the coordinate using the rotation matrix
                indextransformed_coord = R_z @ indexcoord
                wristtransformed_coord = R_z @ wristcoord
                return indextransformed_coord, wristtransformed_coord
            except Exception as e:
                print("Error processing hand landmarks:", e)
                continue
    return None, None
def get_hand_angles(indexPoint, wristPoint):
    indexx = indexPoint[0]
    indexy = indexPoint[1]
    wristx = wristPoint[0]
    wristy = wristPoint[1]
    # Calculate the angle (in radians) between the wrist and index finger relative to the x-axis.
    theta = math.atan((indexy - wristy)/(indexx - wristx))
    if (indexy - wristy < 0):
        theta = math.pi - theta
    
    # Compute the rotation angle needed to align this line with the y-axis.
    rotation_angle = (math.pi / 2) - theta

    # Convert the angle to degrees for readability.
    rotation_angle_deg = (math.degrees(rotation_angle)) -90
    return abs(rotation_angle_deg)

    



try:
    none_counter = 0  # tracks consecutive frames without hand detection
    stable_count = 0  # counts consecutive frames with stable hand coordinates
    hand_counter = 0
    prev_hand_coord = None  # holds the previous hand coordinate for stability comparison
    state = "home"

    while True:
        frames = pipeline.poll_for_frames()
        if not frames:
            continue

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        indexpoint_3d_mm, wristpoint_3d_mm = get_hand_coords(color_frame, depth_frame)
        

        #if indexpoint_3d_mm is not None and wristpoint_3d_mm is not None:
            # angle = get_hand_angles(indexpoint_3d_mm, wristpoint_3d_mm)
        #    get_hand_angles(indexpoint_3d_mm, wristpoint_3d_mm)

        # If no hand is detected, reset stability and count missing frames.
        if indexpoint_3d_mm is None:
            none_counter += 1
            stable_count = 0
            prev_hand_coord = None
            if none_counter >= 10:
                print("No hand detected for 10 frames. Sending robot home.")
                send_coords(home)
                prev_hand_coord = None
                state = "home"
                none_counter = 0
                current_coords = 0  # reset after sending home
                hand_counter = 0
            continue
        else:
            none_counter = 0
        if state=="hand":
            continue
        # Check hand coordinate stability using point_3d_mm.
        if prev_hand_coord is None:
            prev_hand_coord = indexpoint_3d_mm
            stable_count = 1
        else:
            diff = np.linalg.norm(np.array(indexpoint_3d_mm) - np.array(prev_hand_coord))
            if diff <= 10:  # 10 mm = 1 cm threshold
                stable_count += 1
            else:
                stable_count = 1  # reset if the hand moves more than 5cm
            prev_hand_coord = indexpoint_3d_mm

        # Only proceed if we have 10 consecutive stable frames.
        if stable_count < 10:
            continue

        # Once stable for 10 frames, get the robot's current coordinates.
        endEffectorCoords = get_coords()
        if endEffectorCoords is None:
            print("Robot did not return coordinates")
            continue

        end_effector = endEffectorCoords[:3]
        euler_angles = home[3:]

        # Transform the camera coordinates to the robot's coordinate system.
        base_coords = transform_camera_to_robot(indexpoint_3d_mm, end_effector, euler_angles, angles_in_degrees=True)
        print(indexpoint_3d_mm)
        target_coords = np.concatenate((base_coords, euler_angles))
        #turn = get_hand_angles(indexpoint_3d_mm, wristpoint_3d_mm)
        #rz = home[5]
        #if rz + turn >= 170:
        #    rz -= turn
        #else:
        #    rz +=turn
        #target_coords[5] = rz
        send_coords(target_coords)
        time.sleep(2)    
            #send_gripper_command(0, 50)
            #time.sleep(4) 
        send_coords(home)       
        state = "home"
            # Reset the stability counter after sending the move command.
        stable_count = 0
        hand_counter = 0
        none_counter = 0
        prev_hand_coord = None
        print(target_coords)


finally:
    pipeline.stop()
    cv2.destroyAllWindows()
