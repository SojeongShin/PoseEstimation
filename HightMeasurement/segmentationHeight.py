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
C_world = -R_inv @ t  # shape: (3, 1)

# === Load image and (optional) undistort ===
img = cv2.imread("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/ref/shu/measuring.jpg")
h, w = img.shape[:2]
# Uncomment if lens distortion is strong:

# === Mediapipe Selfie Segmentation ===
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
results = segmentor.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# === Threshold segmentation mask ===
CONFIDENCE_THRESHOLD = 0.5
mask = (results.segmentation_mask > CONFIDENCE_THRESHOLD).astype(np.uint8) * 255

ys, xs = np.where(mask > 0)
if len(ys) == 0:
    print("No segmentation detected.")
    exit()

# === Bounding box of mask
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()

# Use center of bounding box as reference column
u_center = int((x_min + x_max) / 2)
u_head, v_head = u_center, y_min
u_feet, v_feet = u_center, y_max

def ray_direction(u, v):
    uv1 = np.array([[u], [v], [1]])
    d_cam = np.linalg.inv(K) @ uv1
    d_world = R_inv @ d_cam
    return d_world / np.linalg.norm(d_world)

# === Compute rays
d_feet = ray_direction(u_feet, v_feet)
s_feet = -C_world[2, 0] / d_feet[2, 0]
feet_world = (C_world + s_feet * d_feet).ravel()

d_head = ray_direction(u_head, v_head)
head_world = (C_world + s_feet * d_head).ravel()

# === Uncorrected height
height_mm = head_world[2] - feet_world[2]

# === Apply z-direction correction factor
dz_mean = (abs(d_head[2, 0]) + abs(d_feet[2, 0])) / 2
height_mm_corrected = height_mm * dz_mean

print(f"Original Height: {height_mm:.1f} mm")
print(f"Corrected Height: {height_mm_corrected:.1f} mm")


# === Print info
print("Camera position (C_world) in mm:")
print("X: {:.2f} mm".format(C_world[0, 0]))
print("Y: {:.2f} mm".format(C_world[1, 0]))
print("Z: {:.2f} mm".format(C_world[2, 0]))
print("Feet position in world coordinates (mm):")
print("X: {:.2f} mm".format(feet_world[0]))
print("Y: {:.2f} mm".format(feet_world[1]))
print("Z: {:.2f} mm".format(feet_world[2]))

# === Visualization
annotated_img = img.copy()
cv2.circle(annotated_img, (u_head, v_head), 5, (255, 0, 0), -1)
cv2.circle(annotated_img, (u_feet, v_feet), 5, (0, 0, 255), -1)
cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
cv2.putText(annotated_img,
            f"Height: {height_mm:.1f} mm",
            (x_min, max(y_min - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imshow("3D Height Estimation", annotated_img)
cv2.imwrite("height_estimation_segmented.jpg", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
