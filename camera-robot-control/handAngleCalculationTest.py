
print("Hello")

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

def get_hand_angles(indexPoint, wristPoint):
    """
    Calculate the rotation angle of the vector formed by the index finger and the wrist point
    relative to the positive X-axis (to the right).
    
    :param indexPoint: Landmark of the index finger (e.g., point 5)
    :param wristPoint: Landmark of the wrist (e.g., point 0)
    :return: Angle in degrees
    """
    # Calculate the vector from wristPoint to indexPoint
    vector = indexPoint - wristPoint
    right = np.array([1, 0])  # Positive X-axis

    # Normalize the vector
    unit_vector = vector / np.linalg.norm(vector)

    # Dot product to get the angle between the vector and the rightward direction
    dot = np.dot(unit_vector, right)

    # Clamp dot product to avoid invalid values due to floating point errors
    angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))

    # Calculate the cross product to determine direction (clockwise or counter-clockwise)
    cross = np.cross(right, unit_vector)
    angle_deg = np.degrees(angle_rad)

    # If the cross product is negative, the angle is clockwise (negative)
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

        # Process hand landmarks
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                if len(lm) > 17:
                    # Get points: index (point 5) and wrist (point 0)
                    wrist = np.array([lm[0].x, lm[0].y])
                    index = np.array([lm[5].x, lm[5].y])

                    # Call the get_hand_angles function
                    angle = get_hand_angles(index, wrist)

                    h, w, _ = image.shape
                    wrist_pt = (int(wrist[0] * w), int(wrist[1] * h))
                    index_pt = (int(index[0] * w), int(index[1] * h))

                    # Draw the vector arrow
                    cv2.arrowedLine(image, wrist_pt, index_pt, (0, 255, 255), 3, tipLength=0.2)
                    cv2.circle(image, wrist_pt, 6, (255, 0, 0), -1)
                    cv2.circle(image, index_pt, 6, (0, 0, 255), -1)

                    # Display the calculated angle
                    cv2.putText(image, f"Angle: {angle:.2f} deg", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Angle - RealSense D405", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
