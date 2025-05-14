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
R = extrinsic_matrix[:, :3]  # (3x3)
t = extrinsic_matrix[:, 3].reshape(3, 1)  # (3x1)

# === Inverse rotation matrix and camera position ===
R_inv = np.linalg.inv(R)
C_world = -R_inv @ t  # 카메라 월드 위치 (3x1)

# === Load image ===
img = cv2.imread("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/mat-pic/stand/jmin/frame_0724.jpg")
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

    def compute_world_point(u, v, s_override=None, force_z0=False):
        uv1 = np.array([[u], [v], [1]])
        d_cam = np.linalg.inv(K) @ uv1
        d_world = R_inv @ d_cam

        if force_z0:
            s = -C_world[2, 0] / d_world[2, 0]
        elif s_override is not None:
            s = s_override
        else:
            raise ValueError("Either force_z0=True or s_override must be provided")

        return (C_world + s * d_world).ravel(), s

    # 발 위치: z=0에 맞추기
    feet_world, s_feet = compute_world_point(u_feet, v_feet, force_z0=True)

    # 머리 위치: 발에서 구한 s값 사용
    head_world, _ = compute_world_point(u_head, v_head, s_override=s_feet)

    # === Height = 머리 z - 발 z
    height_mm = head_world[2] - feet_world[2]
    print(f"Estimated Height: {height_mm:.1f} mm")

    # === Visualization
    annotated_img = img.copy()
    cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.circle(annotated_img, (u_head, v_head), 5, (255, 0, 0), -1)  # 머리: 파란색
    cv2.circle(annotated_img, (u_feet, v_feet), 5, (0, 0, 255), -1)  # 발: 빨간색
    cv2.putText(annotated_img,
                f"Estimated Height: {height_mm:.1f} mm",
                (x_min, max(y_min - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

    # === Show and save result
    cv2.imshow("Estimated Height", annotated_img)
    cv2.imwrite("output.jpg", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Pose not detected.")
