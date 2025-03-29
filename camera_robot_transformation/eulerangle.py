import cv2
import numpy as np
import math
import mediapipe as mp
import pyrealsense2 as rs
import time
from pymycobot import MyCobot280Socket

mc = MyCobot280Socket("10.42.0.1", 5005)


'''
# --- Raspberry Pi Connection Details ---
PI_IP = "10.42.0.1"      # Change to your Pi's IP address
SEND = 5005         # Port for sending target angles (e.g. hand coordinates)
RECEIVE = 5006      # Port for receiving current joint angles



def send_coords_to_pi(coords):
    """
    Packages a list of 6 floats into a struct and sends it to the server.
    """
    if len(coords) != 6:
        raise ValueError("coords must contain exactly 6 elements")
    
    # Pack the 6 floats into a binary structure
    data = struct.pack("6f", *coords)
    
    # Create a TCP socket and send the data
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((PI_IP, SEND))
        sock.sendall(data)
        # Optionally, receive the server's response
        response = sock.recv(1024)
        print("Server response:", response.decode())

def get_coords_from_pi():
    """
    Connects to the Raspberry Pi on RECEIVE, sends a GET command, and
    receives 6 floats representing the current joint angles.
    """
    COMMAND_GET = b"GET_COORDS"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((PI_IP, RECEIVE))
            sock.sendall(COMMAND_GET)
            data = sock.recv(24)  # Expecting 24 bytes (6 floats)
            if len(data) != 24:
                print("Error: Expected 24 bytes for joint coords, got", len(data))
                return None
            coords = struct.unpack("6f", data)
            return coords
    except Exception as e:
        print("Error in get_coords_from_pi:", e)
        return None
'''

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
    """
    Transforms a point from the camera coordinate system to the robot base frame.
    
    This function applies a fixed axis remapping so that:
      - The camera's y coordinate contributes to the robot's x,
      - The camera's z coordinate contributes to the robot's y, and
      - The camera's x coordinate contributes to the robot's z.
    
    After the full transformation, the y and z changes are multiplied by -1
    before adding the end effector's translation.
    
    The full transformation is:
       P_robot = T_end_effector + [dx, -dy, -dz]
       where [dx, dy, dz] = R_end_effector * (R_fixed * P_camera)
    
    Parameters:
      - camera_coords: (x, y, z) measured by the camera (in mm).
      - end_effector_coords: (X, Y, Z) position of the robot's end effector (in mm).
      - euler_angles: (roll, pitch, yaw) of the end effector.
      - angles_in_degrees: Set True if Euler angles are given in degrees.
      
    Returns:
      - robot_coords: The transformed (x, y, z) coordinates in the robot base frame.
    """
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
    
    camera_vec = np.array([[x_c], [y_c], [z_c]])
    
    transformed_change = R_ee @ (R_fixed @ camera_vec)
    
    # Multiply y and z changes by -1 before adding translation.
    
    robot_vec = np.array([[X_ee], [Y_ee], [Z_ee]]) + transformed_change
    
    return robot_vec.flatten()

def get_hand_coords(color_frame, depth_frame):
    """
    Processes the provided color and depth frames to compute the 3D coordinates 
    (in millimeters) of the index finger tip using the same logic as before.
    
    Parameters:
        color_frame: The RealSense color frame.
        depth_frame: The RealSense depth frame.
        
    Returns:
        tuple: (point_3d_mm, pixel_x, pixel_y, color_image)
               point_3d_mm is a list [x, y, z] in mm if a hand is detected,
               pixel_x and pixel_y are the pixel coordinates of the index finger tip,
               and color_image is the NumPy array of the color frame.
               If no hand is detected, point_3d_mm, pixel_x, and pixel_y are None.
    """
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

# Initialize MediaPipe Hands and Drawing modules
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
    home = [50, -64, 420, -90, 0, -90]
    mc.sync_send_coords(home, 10, 0, 0)
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # Get hand coordinates from the current frames
        point_3d_mm, pixel_x, pixel_y, color_image = get_hand_coords(color_frame, depth_frame)
        offsetx = 0
        offsety = 0
        offsetz = 0
        point_3d_mm = [point_3d_mm[0] + offsetx,
               point_3d_mm[1] + offsety,
               point_3d_mm[2] + offsetz]
        # If a hand is detected, process and annotate the image
        if point_3d_mm is not None:
            '''
            # Draw the MediaPipe hand landmarks on the color image
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            '''
            # Compute the transformed robot base coordinates using fixed end effector values


            coords = mc.get_coords()
            print(coords)
            
            end_effector = coords[:3]

            euler_angles = coords[3:]


            base_coords = transform_camera_to_robot(point_3d_mm, end_effector, euler_angles, angles_in_degrees=True)

            target_coords = base_coords + euler_angles
            
            mc.clear_error_information()

            mc.sync_send_coords(target_coords, 10, 0, 0)

            error = mc.get_error_information()

            if(error == 32):
                continue
                    
            mc.clear_error_information()

            time.sleep(5)
            mc.sync_send_coords(home, 10, 0, 0)


        '''
            # Prepare annotation texts
            text1 = f"3D: {np.array(point_3d_mm).round(1)} mm"
            text2 = f"Robot Base: {np.array(robot_coords).round(1)} mm"
            cv2.putText(color_image, text1, (pixel_x, pixel_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, text2, (pixel_x, pixel_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # If no hand is detected, display a message
            cv2.putText(color_image, "No hand detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Hand Tracking", color_image)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit when ESC is pressed
            break
        '''
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
