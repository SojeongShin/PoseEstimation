import cv2
import numpy as np
import mediapipe as mp

# === Load image ===
img = cv2.imread("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/rotated_frames/stand/stand-sj-r.jpg")
h, w = img.shape[:2]

# === Run Mediapipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if results.pose_landmarks:
    # Bounding box from landmarks
    y_coords = [lm.y * h for lm in results.pose_landmarks.landmark]
    y_min = int(min(y_coords))  # top of head (likely eyes)
    y_max = int(max(y_coords))  # bottom (feet or below)

    # 픽셀 거리
    pixel_height = y_max - y_min

    # 픽셀 길이를 mm로 환산하는 scale은 사람 위치나 거리 따라 달라짐 → 고정되지 않음
    print(f"[픽셀 기반 추정] Bounding box height: {pixel_height} pixels")

    # 시각화
    annotated_img = img.copy()
    cv2.rectangle(annotated_img, (0, y_min), (w, y_max), (0, 255, 0), 2)
    cv2.putText(annotated_img, f"Height(px): {pixel_height}",
                (20, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

    # Save and show
    cv2.imwrite("pixel_based_height.jpg", annotated_img)
    cv2.imshow("Pixel Height Only", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Pose not detected.")
