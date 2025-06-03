import os
import cv2
import mediapipe as mp
import numpy as np

# 설정값
real_arm_length_mm = 260
head_to_eye_offset_mm = 120

# 폴더 경로
input_folder = '/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/ref/sy'
output_folder = os.path.join(input_folder, 'prior_output')
os.makedirs(output_folder, exist_ok=True)

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# 모든 이미지 반복
for filename in os.listdir(input_folder):
    if not filename.lower().endswith('.jpg'):
        continue

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read {filename}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print(f"❌ No pose detected in {filename}")
        continue

    landmarks = results.pose_landmarks.landmark

    def get_y(landmark): return landmark.y
    def get_point(landmark): return np.array([landmark.x, landmark.y])

    # === 팔 길이 계산
    l_shoulder = get_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
    l_elbow    = get_point(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
    l_wrist    = get_point(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
    mp_arm_length = np.linalg.norm(l_shoulder - l_elbow)

    if mp_arm_length == 0:
        print(f"⚠️ Zero arm length in {filename}, skipping.")
        continue

    scale = real_arm_length_mm / mp_arm_length

    # === y축 거리
    y_eye = get_y(landmarks[mp_pose.PoseLandmark.LEFT_EYE])
    y_foot = get_y(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])
    mp_y_diff = abs(y_foot - y_eye)

    estimated_length_mm = mp_y_diff * scale
    total_estimated_height_mm = estimated_length_mm + head_to_eye_offset_mm

    # === 시각화
    h, w = image.shape[:2]
    eye_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_EYE].x * w),
              int(landmarks[mp_pose.PoseLandmark.LEFT_EYE].y * h))
    foot_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * w),
               int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * h))

    cv2.circle(image, eye_px, 6, (0, 255, 0), -1)
    cv2.circle(image, foot_px, 6, (0, 0, 255), -1)
    cv2.line(image, eye_px, foot_px, (255, 0, 0), 2)

    text = f"Height: {total_estimated_height_mm:.1f} mm"
    cv2.putText(image, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

    # === 저장
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, image)
    print(f"✅ Saved: {save_path}")
