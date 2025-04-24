import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

def flip_x(landmark):
    return np.array([1.0 - landmark.x, landmark.y])

def calculate_angle(vector):
    right = np.array([1, 0])  # Positive X-axis
    unit_vector = vector / np.linalg.norm(vector)
    dot = np.dot(unit_vector, right)
    angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
    cross = np.cross(right, unit_vector)
    angle_deg = np.degrees(angle_rad)
    if cross < 0:
        angle_deg = -angle_deg
    return angle_deg

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                if len(lm) > 17:
                    # Flip X-coordinates
                    p0 = flip_x(lm[0])
                    p17 = flip_x(lm[17])
                    p5 = flip_x(lm[5])

                    mid = (p0 + p17) / 2
                    vector = p5 - mid
                    angle = calculate_angle(vector)

                    h, w, _ = image.shape
                    start_point = (int(mid[0] * w), int(mid[1] * h))
                    end_point = (int(p5[0] * w), int(p5[1] * h))

                    # Draw vector arrow
                    cv2.arrowedLine(image, start_point, end_point, (0, 255, 255), 3, tipLength=0.2)

                    # Draw key points
                    cv2.circle(image, start_point, 5, (255, 0, 0), -1)
                    cv2.circle(image, end_point, 5, (0, 0, 255), -1)

                    # Draw angle text
                    cv2.putText(image, f"Angle: {angle:.2f} deg", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Vector & Angle (RealSense D405)', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
