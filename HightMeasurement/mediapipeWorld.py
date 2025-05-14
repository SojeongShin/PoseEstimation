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


# === Inverse rotation matrix ===
R_inv = np.linalg.inv(R)

# === Camera position in world coordinates ===
C_world = -R_inv @ t  # 3x1

# === Mediapipe init ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# === Load image ===
img = cv2.imread("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/mat-pic/stand/stand-sj-back.jpg")
h, w = img.shape[:2]

# === Run pose estimation ===
results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if results.pose_landmarks:
    # left foot index = 32
    lm = results.pose_landmarks.landmark[28] 
    u, v = int(lm.x * w), int(lm.y * h)

    # Create homogeneous pixel vector
    uv1 = np.array([[u], [v], [1]])

    # Compute direction vector in camera coordinate
    d_cam = np.linalg.inv(K) @ uv1
    d_world = R_inv @ d_cam

    # Solve for scale s using Z=0 (ground plane)
    s = -C_world[2, 0] / d_world[2, 0]

    # Compute 3D world coordinate
    X_world = C_world + s * d_world

    print("Right ankle 3D world position (mm):", X_world.ravel())
else:
    print("Pose not detected.")

# === Draw point on image ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Draw pose landmarks on image
if results.pose_landmarks:
    annotated_image = img.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
    )

    cv2.imshow("Pose Skeleton", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
