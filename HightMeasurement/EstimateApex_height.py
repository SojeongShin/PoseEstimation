import cv2
import numpy as np
import mediapipe as mp

# === Load intrinsic parameters ===
intrin = np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/cameraCali/intrinsic_calibration_result.npz")
K = intrin["K"]
dist = intrin["dist"]

# === Load extrinsic parameters ===
extrin = np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/final_extrinsic_calibration.npz")
extrinsic_matrix = extrin["extrinsic_matrix"]
R = extrinsic_matrix[:, :3]
t = extrinsic_matrix[:, 3].reshape(3, 1)

# === Inverse rotation and camera position ===
R_inv = np.linalg.inv(R)
C_world = -R_inv @ t

# === Load image ===
img = cv2.imread("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/mat-pic/stand/stand-sj-Te.jpg")
h, w = img.shape[:2]

# === Run Mediapipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark

    # === Estimate feet location using both ankles
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]
    u_feet = int(((left_ankle.x + right_ankle.x) / 2) * w)
    v_feet = int(((left_ankle.y + right_ankle.y) / 2) * h)

    # === Estimate face center using eyes, ears, nose
    face_ids = [0, 1, 2, 3, 4]  # nose, L/R eye, L/R ear
    u_face = int(np.mean([landmarks[i].x for i in face_ids]) * w)
    v_face = int(np.mean([landmarks[i].y for i in face_ids]) * h)

    # === Estimate apex as 12% upward from facial keypoints
    pixel_face_height = v_feet - v_face
    v_apex = max(int(v_face - 0.12 * pixel_face_height), 0)
    u_apex = u_face

    def ray_direction(u, v):
        uv1 = np.array([[u], [v], [1]])
        d_cam = np.linalg.inv(K) @ uv1
        d_world = R_inv @ d_cam
        return d_world / np.linalg.norm(d_world)

    # Feet ray
    d_feet = ray_direction(u_feet, v_feet)
    s_feet = -C_world[2, 0] / d_feet[2, 0]
    feet_world = C_world + s_feet * d_feet

    # Apex ray
    d_apex = ray_direction(u_apex, v_apex)
    apex_world = C_world + s_feet * d_apex
    height_mm = apex_world[2]  # feet z = 0, so apex z = height
    height_mm = float(height_mm)

    print(f"Estimated Height (Apex): {height_mm:.1f} mm")

    # === Visualization
    annotated_img = img.copy()
    cv2.circle(annotated_img, (u_feet, v_feet), 5, (0, 0, 255), -1)    # Feet (red)
    cv2.circle(annotated_img, (u_apex, v_apex), 5, (0, 255, 255), -1)  # Apex (yellow)
    cv2.putText(annotated_img,
                f"Height (Apex): {height_mm:.1f} mm",
                (u_apex, max(v_apex - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Estimated Apex Height", annotated_img)
    cv2.imwrite("estimated_apex_height.jpg", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # === Draw pose 7

else:
    print("Pose not detected.")

for idx, lm in enumerate(landmarks):
    u = int(lm.x * w)
    v = int(lm.y * h)
    cv2.circle(annotated_img, (u, v), 3, (0, 255, 0), -1)
    cv2.putText(annotated_img, str(idx), (u + 5, v - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)

