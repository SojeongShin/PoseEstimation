import cv2
import numpy as np

# === Load intrinsic parameters ===
intrin = np.load("/Users/sojeongshin/Documents/GitHub/PoseEstimation/HightMeasurement/cameraCali/intrinsic_calibration_result.npz")
K = intrin["K"]
dist = intrin["dist"]

# === Load extrinsic parameters ===
extrin = np.load("/Users/sojeongshin/Documents/GitHub/PoseEstimation/HightMeasurement/final_extrinsic_calibration.npz")
extrinsic_matrix = extrin["extrinsic_matrix"]
R = extrinsic_matrix[:, :3]
t = extrinsic_matrix[:, 3].reshape(3, 1)

# === Inverse rotation and camera position ===
R_inv = np.linalg.inv(R)
C_world = -R_inv @ t

# === Load image
img = cv2.imread("/Users/sojeongshin/Documents/GitHub/PoseEstimation/HightMeasurement/ref/sj/O.jpg")
clone = img.copy()
clicks = []

# === Mouse callback
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
        clicks.append((x, y))
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click Head and Feet", clone)

cv2.imshow("Click Head and Feet", clone)
cv2.setMouseCallback("Click Head and Feet", mouse_callback)

# Wait for two clicks
while True:
    cv2.imshow("Click Head and Feet", clone)
    if len(clicks) == 2:
        cv2.destroyAllWindows()
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break

(u_head, v_head), (u_feet, v_feet) = sorted(clicks, key=lambda pt: pt[1])  # 머리가 위쪽

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

# === Height (z축 기준)
height_mm = head_world[2] - feet_world[2]

# === Print results
print(f"Height: {height_mm:.1f} mm")

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
cv2.line(annotated_img, (u_head, v_head), (u_feet, v_feet), (0, 255, 0), 2)
cv2.putText(annotated_img,
            f"Height: {height_mm:.1f} mm",
            (u_head, max(v_head - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imshow("Height Estimation", annotated_img)
cv2.imwrite("height_estimation_manual.jpg", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
