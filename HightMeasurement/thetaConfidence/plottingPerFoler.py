import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# === Load intrinsic and extrinsic parameters ===
intrin = np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/cameraCali/intrinsic_calibration_result.npz")
K = intrin["K"]
dist = intrin["dist"]

extrin = np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/final_extrinsic_calibration.npz")
extrinsic_matrix = extrin["extrinsic_matrix"]
R = extrinsic_matrix[:, :3]
t = extrinsic_matrix[:, 3].reshape(3, 1)
R_inv = np.linalg.inv(R)
C_world = -R_inv @ t

# === Helper functions ===
def ray_direction(u, v):
    uv1 = np.array([[u], [v], [1]])
    d_cam = np.linalg.inv(K) @ uv1
    d_world = R_inv @ d_cam
    return d_world / np.linalg.norm(d_world)

def compute_height_and_angle(u_head, v_head, u_feet, v_feet):
    d_feet = ray_direction(u_feet, v_feet)
    s_feet = -C_world[2, 0] / d_feet[2, 0]
    feet_world = (C_world + s_feet * d_feet).ravel()

    d_head = ray_direction(u_head, v_head)
    head_world = (C_world + s_feet * d_head).ravel()

    height = head_world[2] - feet_world[2]

    # Angle between head ray and z-axis
    z_axis = np.array([[0], [0], [1]])
    cos_theta = float((d_head.T @ z_axis) / np.linalg.norm(d_head))
    theta_deg = np.degrees(np.arccos(cos_theta))

    return height, theta_deg

# === Manual clicking handler ===
def get_clicks(img):
    clone = img.copy()
    clicks = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Click Head and Feet", clone)

    cv2.imshow("Click Head and Feet", clone)
    cv2.setMouseCallback("Click Head and Feet", mouse_callback)

    while True:
        cv2.imshow("Click Head and Feet", clone)
        if len(clicks) == 2:
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(1) & 0xFF == 27:
            break

    return sorted(clicks, key=lambda pt: pt[1])  # 머리가 위쪽

# === Main Loop ===
image_dir = "/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/ref/ab"  # 이미지 폴더 경로
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

theta_list = []
height_list = []

for idx, fname in enumerate(image_files):
    print(f"[{idx+1}/{len(image_files)}] Processing: {fname}")
    img_path = os.path.join(image_dir, fname)
    img = cv2.imread(img_path)
    (u_head, v_head), (u_feet, v_feet) = get_clicks(img)

    height, theta = compute_height_and_angle(u_head, v_head, u_feet, v_feet)
    print(f" -> θ: {theta:.2f} deg | Height: {height:.1f} mm")

    theta_list.append(theta)
    height_list.append(height)

# === Sort data by theta
theta_height_pairs = sorted(zip(theta_list, height_list), key=lambda x: x[0])
sorted_theta, sorted_height = zip(*theta_height_pairs)

# === Plotting
plt.figure(figsize=(8, 5))
plt.plot(sorted_theta, sorted_height, 'o-r', linewidth=2)
plt.xlabel("Theta (deg)")
plt.ylabel("Measured Height (mm)")
plt.title("Measured Height vs. Angle θ")
plt.grid(True)
plt.tight_layout()
plt.savefig("theta_vs_height.png")
plt.show()
