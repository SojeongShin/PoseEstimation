import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import os
import glob

# === Load intrinsic & extrinsic parameters ===
intrin = np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/cameraCali/intrinsic_calibration_result.npz")
K = intrin["K"]
dist = intrin["dist"]

extrin = np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/final_extrinsic_calibration.npz")
extrinsic_matrix = extrin["extrinsic_matrix"]
R = extrinsic_matrix[:, :3]
t = extrinsic_matrix[:, 3].reshape(3, 1)

R_inv = np.linalg.inv(R)
C_world = -R_inv @ t  # camera center in world coordinates

# === Prepare model ===
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights).eval()
preprocess = weights.transforms()

# === Paths ===
input_dir = "/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/ref/jm"
output_dir = os.path.join(input_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# === Iterate over all .png files in input folder ===
image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

for img_path in image_paths:
    print(f"Processing: {os.path.basename(img_path)}")
    
    img_cv = cv2.imread(img_path)
    img_pil = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(img_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    mask = output.argmax(0).byte().cpu().numpy()
    person_mask = (mask == 15).astype(np.uint8) * 255
    mask_resized = cv2.resize(person_mask, (img_cv.shape[1], img_cv.shape[0]))

    # === Find topmost head y (with min width) ===
    min_head_width = 35
    v_head = None
    for y in range(mask_resized.shape[0]):
        x_coords = np.where(mask_resized[y] == 255)[0]
        if len(x_coords) >= min_head_width:
            v_head = y
            u_head = int(np.median(x_coords))
            break

    if v_head is None:
        print(f"⚠️ Skipped: no valid head found in {os.path.basename(img_path)}")
        continue

    ys, xs = np.where(mask_resized == 255)
    v_feet = ys.max()
    u_feet = u_head

    def ray_direction(u, v):
        uv1 = np.array([[u], [v], [1]])
        d_cam = np.linalg.inv(K) @ uv1
        d_world = R_inv @ d_cam
        return d_world / np.linalg.norm(d_world)

    d_feet = ray_direction(u_feet, v_feet)
    s_feet = -C_world[2, 0] / d_feet[2, 0]
    feet_world = (C_world + s_feet * d_feet).ravel()

    d_head = ray_direction(u_head, v_head)
    head_world = (C_world + s_feet * d_head).ravel()

    height_mm = head_world[2] - feet_world[2]
    z_axis = np.array([[0], [0], [1]])
    cos_theta = float((d_head.T @ z_axis) / np.linalg.norm(d_head))
    theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    pixel_height_mm = height_mm / (v_feet - v_head)

    print(f"  → Height: {height_mm:.1f} mm | θ: {theta_deg:.2f}° | 1px ≈ {pixel_height_mm:.2f} mm")

    # === Visualization ===
    annotated = img_cv.copy()
    cv2.circle(annotated, (u_head, v_head), 5, (255, 0, 0), -1)
    cv2.circle(annotated, (u_feet, v_feet), 5, (0, 0, 255), -1)
    cv2.line(annotated, (u_head, v_head), (u_feet, v_feet), (0, 255, 0), 2)
    cv2.putText(annotated, f"Height: {height_mm:.1f} mm",
                (u_head, max(v_head - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.line(annotated, (0, v_head), (annotated.shape[1], v_head), (0, 0, 255), 1)

    # === Save output ===
    filename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_dir, f"height_estimation_{filename}.jpg")
    cv2.imwrite(save_path, annotated)

print("✅ All images processed and saved in:", output_dir)
