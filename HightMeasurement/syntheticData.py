import numpy as np
import torch
import smplx
import trimesh
import pyrender
import cv2

# === 0. 설정 ===
target_height_mm = 1600  # 원하는 신장(mm)

# === 1. Load camera parameters ===
K = np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/cameraCali/intrinsic_calibration_result.npz")["K"]
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

Rt = np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/final_extrinsic_calibration.npz")["extrinsic_matrix"]
R, t = Rt[:, :3], Rt[:, 3].reshape(3, 1)
pose = np.eye(4)
pose[:3, :3] = R
pose[:3, 3] = t.squeeze()
pose = np.linalg.inv(pose)

# === 2. Load SMPL model ===
model = smplx.create(model_path='/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/SMPL_NEUTRAL.pkl', model_type='smpl', gender='neutral', use_pca=False)

# === 3. 신장에 맞는 beta0 찾기 ===
def find_beta0_for_target_height(target_height_mm, model, tolerance=10.0):
    for beta0 in np.linspace(-3.0, 3.0, 100):
        betas = torch.zeros([1, 10])
        betas[0, 0] = beta0
        output = model(betas=betas, return_verts=True)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        height = vertices[:, 1].max() - vertices[:, 1].min()

        # scale fix if height too small
        if height < 10:  # likely in meters
            height *= 1000

        if abs(height - target_height_mm) < tolerance:
            return beta0, height
    return None, None


beta0, height_found = find_beta0_for_target_height(target_height_mm, model)
if beta0 is None:
    raise ValueError("Target height not found in beta range.")

print(f"[INFO] Found beta0 = {beta0:.3f} for height ≈ {height_found:.1f} mm")

betas = torch.zeros([1, 10])
betas[0, 0] = beta0
output = model(betas=betas, return_verts=True)
vertices = output.vertices.detach().cpu().numpy().squeeze()
faces = model.faces

# === 4. 렌더링 ===
mesh = trimesh.Trimesh(vertices, faces)
scene = pyrender.Scene()
scene.add(pyrender.Mesh.from_trimesh(mesh))

camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
scene.add(camera, pose=pose)

light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=pose)

r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
rgb, _ = r.render(scene)
cv2.imwrite("output_rendered_height.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

print("[INFO] Saved rendered image for height:", height_found)
