import cv2
import numpy as np

# === STEP 1: Load intrinsic parameters ===
intrinsic = np.load('HightMeasurement/cameraCali/intrinsic_calibration_result.npz')
K = intrinsic['K']
dist = intrinsic['dist']

# === STEP 2: Load image and get 4 image points from clicks ===
image = cv2.imread('/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/cameraCali/rotated_frames/frame_1091.jpg')  # 📸 바닥 매트가 있는 이미지
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked: ({x}, {y})")
        clicked_points.append([x, y])
        cv2.circle(image, (x, y), 6, (0, 255, 0), -1)
        cv2.imshow("Click 4 Mat Points", image)

print("🖱️ Please click 4 known mat points in order.")
cv2.imshow("Click 4 Mat Points", image)
cv2.setMouseCallback("Click 4 Mat Points", mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(clicked_points) != 4:
    raise ValueError("❌ You must click exactly 4 points!")

image_points = np.array(clicked_points, dtype=np.float32)

# === STEP 3: Define corresponding 3D mat points (unit: mm) ===
object_points = np.array([
    [0, 0, 0],
    [1830, 0, 0],
    [1260, 700, 0],
    [500, 700, 0]
], dtype=np.float32)

# === STEP 4: Solve PnP ===
success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)

if not success:
    raise RuntimeError("❌ solvePnP failed!")

print("✅ solvePnP succeeded.")
print("Rotation Vector (rvec):\n", rvec)
print("Translation Vector (tvec):\n", tvec)

R, _ = cv2.Rodrigues(rvec)
extrinsic = np.hstack((R, tvec))
print("Extrinsic Matrix [R|t]:\n", extrinsic)

# === STEP 5: Save result ===
np.savez('extrinsic_calibration_result.npz', extrinsic_matrix=extrinsic, rvec=rvec, tvec=tvec)
print("💾 Saved to 'extrinsic_calibration_result.npz'")

# === STEP 6: Reprojection Error ===
projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
projected = projected.squeeze()
errors = np.linalg.norm(image_points - projected, axis=1)
print("📐 Reprojection errors per point (pixels):", errors)
print("📏 Mean reprojection error:", np.mean(errors))

# === STEP 7: Visual check ===
image_vis = cv2.imread('/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/cameraCali/rotated_frames/frame_1091.jpg')
for (u, v) in image_points:
    cv2.circle(image_vis, (int(u), int(v)), 5, (0, 255, 0), -1)  # original (green)
for (u, v) in projected:
    cv2.circle(image_vis, (int(u), int(v)), 5, (0, 0, 255), -1)  # projected (red)

cv2.imshow("Green: Original / Red: Projected", image_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
