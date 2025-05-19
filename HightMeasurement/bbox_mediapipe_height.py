import cv2
import numpy as np
import mediapipe as mp

# === Load intrinsic parameters ===
intrin = np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/cameraCali/intrinsic_calibration_result.npz")
K = intrin["K"]
dist = intrin["dist"]

# === Load extrinsic parameters ===
extrin = np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/final_extrinsic_calibration.npz")
extrinsic_matrix = extrin["extrinsic_matrix"]  # shape: (3, 4)

# Split into R and t
R = extrinsic_matrix[:, :3]
t = extrinsic_matrix[:, 3].reshape(3, 1)

# === Inverse rotation and camera position ===
R_inv = np.linalg.inv(R)
C_world = -R_inv @ t  # 카메라 월드 좌표 (3x1)

# === Load image ===
img = cv2.imread("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/mat-pic/stand/stand-sj-Te.jpg")
h, w = img.shape[:2]

# === Run Mediapipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if results.pose_landmarks:
    # === Bounding box from all landmarks ===
    x_coords = [lm.x * w for lm in results.pose_landmarks.landmark]
    y_coords = [lm.y * h for lm in results.pose_landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    # 발: bbox 하단 중앙 / 머리: bbox 상단 중앙
    u_feet, v_feet = (x_min + x_max) // 2, y_max
    u_head, v_head = (x_min + x_max) // 2, y_min

    def ray_direction(u, v):
        uv1 = np.array([[u], [v], [1]])
        d_cam = np.linalg.inv(K) @ uv1
        d_world = R_inv @ d_cam
        d_world = d_world / np.linalg.norm(d_world)  # 정규화 (선택적)
        return d_world

    # 발 ray로 s 계산 (z=0에서 교차하도록)
    d_feet = ray_direction(u_feet, v_feet)
    s_feet = -C_world[2, 0] / d_feet[2, 0]

    # 머리 ray
    d_head = ray_direction(u_head, v_head)

    # 머리의 월드 좌표
    head_world = C_world + s_feet * d_head
    Z_head = head_world[2]

    # 발은 z=0이므로 키 = 머리의 z
    height_mm = float(Z_head)
    print(f"Estimated Height (using ray + camera height): {height_mm:.1f} mm")
    

    # === Visualization
    annotated_img = img.copy()
    cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.circle(annotated_img, (u_head, v_head), 5, (255, 0, 0), -1)  # 머리
    cv2.circle(annotated_img, (u_feet, v_feet), 5, (0, 0, 255), -1)  # 발
    cv2.putText(annotated_img,
                f"Estimated Height: {height_mm:.1f} mm",
                (x_min, max(y_min - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

    # === Show and save
    cv2.imshow("Estimated Height", annotated_img)
    cv2.imwrite("output.jpg", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Pose not detected.")
