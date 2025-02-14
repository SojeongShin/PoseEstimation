import os
import cv2
import mediapipe as mp
import math
import time
import matplotlib.pyplot as plt
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

PLM = mp_pose.PoseLandmark

# 오른쪽 측면 랜드마크 (optional if you want to manually connect only the right side)
RIGHT_LANDMARKS = [
    PLM.RIGHT_SHOULDER,  # 12
    PLM.RIGHT_ELBOW,     # 14
    PLM.RIGHT_WRIST,     # 16
    PLM.RIGHT_INDEX,     # 20
    PLM.RIGHT_HIP,       # 24
    PLM.RIGHT_KNEE,      # 26
    PLM.RIGHT_ANKLE,     # 28
    PLM.RIGHT_HEEL,      # 30
    PLM.RIGHT_FOOT_INDEX # 32
]

# 오른쪽 측면 연결 관계 (optional if you want to manually connect only the right side)
RIGHT_CONNECTIONS = [
    (PLM.RIGHT_SHOULDER, PLM.RIGHT_ELBOW),
    (PLM.RIGHT_ELBOW, PLM.RIGHT_WRIST),
    (PLM.RIGHT_SHOULDER, PLM.RIGHT_HIP),
    (PLM.RIGHT_HIP, PLM.RIGHT_KNEE),
    (PLM.RIGHT_KNEE, PLM.RIGHT_ANKLE),
    (PLM.RIGHT_ANKLE, PLM.RIGHT_HEEL),
    (PLM.RIGHT_HEEL, PLM.RIGHT_FOOT_INDEX),
    (PLM.RIGHT_WRIST, PLM.RIGHT_INDEX),
]

# Moving average buffer size
BUFFER_SIZE = 3
right_index_buffer = [0] * BUFFER_SIZE
right_foot_index_buffer = [0] * BUFFER_SIZE

def mov_avg_filter(n_frames, x_meas):
    """Apply moving average filter with a buffer size."""
    n_frames.pop(0)  # Remove the oldest entry
    n_frames.append(x_meas)  # Add the new entry
    return np.mean(n_frames), n_frames

def draw_right_side_and_angle(image, landmarks, filtered_index_y, filtered_foot_y, angle_offset=15, visibility_th=0.5):
    h, w, _ = image.shape
    white = (255, 255, 255)
    
    hip_lm = landmarks[PLM.RIGHT_HIP.value]
    knee_lm = landmarks[PLM.RIGHT_KNEE.value]
    ankle_lm = landmarks[PLM.RIGHT_ANKLE.value]
    
    angle_text = ""
    dy_text = ""

    # Check visibility before computing angles
    if (hip_lm.visibility > visibility_th and 
        knee_lm.visibility > visibility_th and 
        ankle_lm.visibility > visibility_th):
        
        hip_x, hip_y = hip_lm.x * w, hip_lm.y * h
        knee_x, knee_y = knee_lm.x * w, knee_lm.y * h
        ankle_x, ankle_y = ankle_lm.x * w, ankle_lm.y * h
        
        # Compute knee angle
        angle = compute_angle_knee(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y)
        angle_text = f"Angle: {angle:.0f}°"
        
        # Check if the angle is "nearly straight"
        if (180 - angle_offset) <= angle <= (180 + angle_offset):
            angle_text += " (Nearly Straight)"

            # finger - foot
            dy = (filtered_index_y - filtered_foot_y)*(0.26)  # 픽셀 단위 차이 * cm value
            dy_text = f"Filtered Y diff: {dy:.1f} cm"

#  //////////////////////////////////////////////////////////////////////////////////////////////////


    # Draw text on the screen
    if angle_text:
        cv2.putText(image, angle_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, white, 2, cv2.LINE_AA)
    if dy_text:
        cv2.putText(image, dy_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

def compute_angle_knee(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y):
    # Vector from knee to hip
    ux, uy = hip_x - knee_x, hip_y - knee_y
    # Vector from knee to ankle
    vx, vy = ankle_x - knee_x, ankle_y - knee_y
    
    dot = ux * vx + uy * vy
    mag_u = math.sqrt(ux**2 + uy**2)
    mag_v = math.sqrt(vx**2 + vy**2)
    if mag_u * mag_v == 0:
        return 0.0
    cos_theta = dot / (mag_u * mag_v)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))

def main():
    cap = cv2.VideoCapture(0)
    # Adjust resolution and FPS as desired
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 768)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    cap.set(cv2.CAP_PROP_FPS, 27)
    
    # Create directories if they don't exist
    os.makedirs('snapshots', exist_ok=True)
    os.makedirs('./plot_filtered', exist_ok=True)
    
    frame_indices = []
    right_index_ys = []
    right_foot_index_ys = []
    
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, 
                      min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("카메라를 찾을 수 없습니다.")
                continue

            frame_count += 1
            
            # Rotate webcam frame if needed
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Flip horizontally for a selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for Mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h = frame.shape[0]
                
                # Landmarks for Right Index and Right Foot Index
                right_index_lm = landmarks[PLM.RIGHT_INDEX.value]
                right_foot_index_lm = landmarks[PLM.RIGHT_FOOT_INDEX.value]
                
                idx_y = right_index_lm.y * h
                ft_y = right_foot_index_lm.y * h
                
                # Apply moving average filter
                filtered_idx_y, right_index_buffer[:] = mov_avg_filter(right_index_buffer, idx_y)
                filtered_ft_y, right_foot_index_buffer[:] = mov_avg_filter(right_foot_index_buffer, ft_y)
                
                # Store filtered data for plotting later
                frame_indices.append(frame_count)
                right_index_ys.append(filtered_idx_y)
                right_foot_index_ys.append(filtered_ft_y)
                
                # 1) Draw the full skeleton from Mediapipe
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    # mp_pose.POSE_CONNECTIONS,  # you can also use RIGHT_CONNECTIONS if you only want the right side
                    # landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=5),
                    # connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5, circle_radius=7)

                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 255, 255),  # white
                        thickness=7,
                        circle_radius=6
                    ),
                    # keep or customize the connection_drawing_spec as needed
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 0, 0), 
                        thickness=6
                    )
                )
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 0, 0),   # blue
                        thickness=-1,       # try "filled", or a large number if it doesn't work
                        circle_radius=5
                    ),
                    # If you want to avoid re‐drawing connections, set connection_drawing_spec=None
                    # connection_drawing_spec=None
                )
                
                # 2) Draw angle info/text overlays
                draw_right_side_and_angle(frame, landmarks, filtered_idx_y, filtered_ft_y)

            # Save the current frame as an image
            snapshot_path = os.path.join('snapshots', f'frame_{frame_count:04d}.png')
            cv2.imwrite(snapshot_path, frame)

            # Display the frame
            cv2.imshow('Filtered Skeleton', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to quit
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Plot and save filtered Y position data
    plt.figure(figsize=(8, 4))
    plt.plot(frame_indices, right_index_ys, color='blue', label='Filtered Right Index Y')
    plt.xlabel('Frame')
    plt.ylabel('Y position (pixels)')
    plt.title('Filtered Right Index Y vs Frame')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plot_filtered/right_index_filtered.png')
    plt.close()
    
    plt.figure(figsize=(8, 4))
    plt.plot(frame_indices, right_foot_index_ys, color='red', label='Filtered Right Foot Index Y')
    plt.xlabel('Frame')
    plt.ylabel('Y position (pixels)')
    plt.title('Filtered Right Foot Index Y vs Frame')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plot_filtered/right_foot_index_filtered.png')
    plt.close()
    
    print("Filtered data plots saved to ./plot_filtered/")
    print("Screenshots saved to ./snapshots/")

if __name__ == "__main__":
    main()
