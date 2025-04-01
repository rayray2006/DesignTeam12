import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

def main():
    # --- Configure and start the RealSense pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    # Align depth frames to color frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # Retrieve intrinsics for deprojection/projection.
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    
    # --- Setup MediaPipe Hands ---
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color.
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert the color frame to a numpy array.
            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Process the image to detect hands.
            results = hands.process(rgb_image)
            
            # If hand(s) are detected, try processing them.
            if results.multi_hand_landmarks:
                try:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks for visualization.
                        mp_draw.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        h, w, _ = color_image.shape
                        
                        # Retrieve landmarks 0, 5, and 17.
                        lm0 = hand_landmarks.landmark[0]    # Typically the wrist.
                        lm5 = hand_landmarks.landmark[5]
                        lm17 = hand_landmarks.landmark[17]
                        
                        # Convert normalized coordinates to pixel coordinates.
                        x0, y0 = int(lm0.x * w), int(lm0.y * h)
                        x5, y5 = int(lm5.x * w), int(lm5.y * h)
                        x17, y17 = int(lm17.x * w), int(lm17.y * h)
                        
                        # Get the depth (in meters) at each landmark pixel.
                        depth0 = depth_frame.get_distance(x0, y0)
                        depth5 = depth_frame.get_distance(x5, y5)
                        depth17 = depth_frame.get_distance(x17, y17)
                        
                        # Check for valid depth values.
                        if depth0 == 0 or depth5 == 0 or depth17 == 0:
                            continue
                        
                        # Deproject the 2D pixels (with depth) into 3D points.
                        point0 = rs.rs2_deproject_pixel_to_point(intrinsics, [x0, y0], depth0)
                        point5 = rs.rs2_deproject_pixel_to_point(intrinsics, [x5, y5], depth5)
                        point17 = rs.rs2_deproject_pixel_to_point(intrinsics, [x17, y17], depth17)
                        
                        # Convert points to numpy arrays.
                        p0   = np.array(point0)
                        p5   = np.array(point5)
                        p17  = np.array(point17)
                        
                        # Compute the weighted average so that it is 1/4 from point 17 (closer to 17).
                        p_avg = p17 + 0.25 * (p0 - p17)
                        
                        # Compute the vector from landmark 5 to the weighted average point.
                        new_point = p5 - 0.5 * (p_avg - p5)
                        
                        # Print computed 3D coordinates.
                        print("Landmark 5 (3D):", p5)
                        print("Landmark 0 (3D):", p0)
                        print("Landmark 17 (3D):", p17)
                        print("Weighted Average (0,17) (3D):", p_avg)
                        print("Extended Landmark 5 (3D):", new_point)
                        
                        # --- Annotation: Project points to image space ---
                        avg_pixel = rs.rs2_project_point_to_pixel(intrinsics, p_avg.tolist())
                        avg_pixel = (int(avg_pixel[0]), int(avg_pixel[1]))
                        
                        new_point_pixel = rs.rs2_project_point_to_pixel(intrinsics, new_point.tolist())
                        new_point_pixel = (int(new_point_pixel[0]), int(new_point_pixel[1]))
                        
                        # Draw an arrowed line from landmark 5 to the extended point.
                        cv2.arrowedLine(color_image, (x5, y5), new_point_pixel, (0, 255, 0), 2, tipLength=0.1)
                        
                        # Mark landmark 5.
                        cv2.circle(color_image, (x5, y5), 5, (255, 255, 0), -1)
                        cv2.putText(color_image, "Landmark 5", (x5 + 5, y5 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # Mark the weighted average point.
                        cv2.circle(color_image, avg_pixel, 5, (255, 0, 0), -1)
                        cv2.putText(color_image, "Avg (0,17)", (avg_pixel[0] + 5, avg_pixel[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # Mark the new extended point.
                        cv2.circle(color_image, new_point_pixel, 5, (0, 0, 255), -1)
                        cv2.putText(color_image, "Extended 5", (new_point_pixel[0] + 5, new_point_pixel[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Optionally, annotate the vector by labeling its midpoint.
                        mid_x = int((x5 + new_point_pixel[0]) / 2)
                        mid_y = int((y5 + new_point_pixel[1]) / 2)
                        cv2.putText(color_image, "Vector", (mid_x, mid_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except Exception as e:
                    # Catch any errors during hand processing.
                    print("Error processing hand landmarks:", e)
                    cv2.putText(color_image, "Error processing hand", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # No hand detected.
                cv2.putText(color_image, "No hand detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
            # Display the annotated image.
            cv2.imshow("RealSense Hand Tracking", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
