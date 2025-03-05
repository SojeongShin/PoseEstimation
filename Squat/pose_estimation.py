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

cap = cv2.VideoCapture(0)
squat_count = 0
in_squat_position = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # 얼굴 정면 확인
        face_check = (abs(nose.x - left_eye.x) < abs(nose.x - right_eye.x) * 1.5 and
                      abs(nose.x - right_eye.x) < abs(nose.x - left_eye.x) * 1.5)
        
        # 등 일자 확인
        back_check = (abs(left_shoulder.x - left_hip.x) < 50 and
                      abs(right_shoulder.x - right_hip.x) < 50)
        
        # 무릎 각도 계산
        left_knee_angle = calculate_angle([left_hip.x, left_hip.y],
                                          [left_knee.x, left_knee.y],
                                          [left_ankle.x, left_ankle.y])
        right_knee_angle = calculate_angle([right_hip.x, right_hip.y],
                                           [right_knee.x, right_knee.y],
                                           [right_ankle.x, right_ankle.y])
        
        # 스쿼트 감지
        if face_check and back_check and left_knee_angle >= 45 and right_knee_angle >= 45:
            in_squat_position = True
        
        # 스쿼트 초기화 조건 확인 (완전히 선 자세)
        if in_squat_position and left_knee_angle < 10 and right_knee_angle < 10:
            squat_count += 1
            in_squat_position = False
            print(f"Squat Count: {squat_count}")
        
        # 랜드마크 시각화
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.putText(image, f'Squat Count: {squat_count}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Squat Counter', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
