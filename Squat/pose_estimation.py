import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load Video Instead of Webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 768)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
cap.set(cv2.CAP_PROP_FPS, 27)

squat_count = 0
state = 0  # 0 = Standing, 1 = Squatting

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        left_knee_angle = calculate_angle([left_hip.x, left_hip.y],
                                          [left_knee.x, left_knee.y],
                                          [left_ankle.x, left_ankle.y])
        right_knee_angle = calculate_angle([right_hip.x, right_hip.y],
                                           [right_knee.x, right_knee.y],
                                           [right_ankle.x, right_ankle.y])
        
        # start squat
        if state == 0 and left_knee_angle >= 170 and right_knee_angle >= 170:
            state = 0

        # Detect Squat
        elif state == 0 and left_knee_angle <= 125 and right_knee_angle <= 125:
            state = 1  # Move to squat position

        # Detect Squat Completion & Increase Count
        elif state == 1 and left_knee_angle >= 170 and right_knee_angle >= 170:
            state = 0
            squat_count += 1
            print(f"Squat Count: {squat_count}")

        # Display knee angles
        cv2.putText(image, f'{int(left_knee_angle)}°',
                    (int(left_knee.x * image.shape[1]), int(left_knee.y * image.shape[0]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'{int(right_knee_angle)}°',
                    (int(right_knee.x * image.shape[1]), int(right_knee.y * image.shape[0]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.putText(image, f'Squat Count: {squat_count}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Squat Counter', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
