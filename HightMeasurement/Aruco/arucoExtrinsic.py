import cv2
import numpy as np

# Load intrinsics
intrin = np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/cameraCali/intrinsic_calibration_result.npz")
K = intrin["K"]
dist = intrin["dist"]

# Load image and detect
img = cv2.imread("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/Aruco/arucoMat.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
params = cv2.aruco.DetectorParameters()
corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

if ids is None:
    raise ValueError("❌ No markers detected.")

# Set marker size in meters
marker_length = 0.245
vis_img = img.copy()

# Estimate pose for each marker
rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, K, dist)

total_error_px = []
total_error_mm = []

for i, id_val in enumerate(ids.flatten()):
    rvec = rvecs[i]
    tvec = tvecs[i]
    corner_img = corners[i][0]  # shape: (4, 2)

    # Define marker 3D corners (relative to center)
    half = marker_length / 2
    objp = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

    # Reproject 3D corners
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)

    # Draw real corners (green) and projected (red)
    for j in range(4):
        pt_real = tuple(np.round(corner_img[j]).astype(int))
        pt_proj = tuple(np.round(proj[j]).astype(int))
        cv2.circle(vis_img, pt_real, 5, (0,255,0), -1)
        cv2.circle(vis_img, pt_proj, 4, (0,0,255), 2)
        cv2.line(vis_img, pt_real, pt_proj, (255,0,0), 1)

    # Compute reprojection error for this marker
    errors_px = np.linalg.norm(corner_img - proj, axis=1)
    mean_px = np.mean(errors_px)

    # Convert to mm (approximate): use side length in pixels
    pixel_len = np.linalg.norm(corner_img[0] - corner_img[1])
    mm_per_pixel = (marker_length * 1000) / pixel_len
    mean_mm = mean_px * mm_per_pixel

    total_error_px.append(mean_px)
    total_error_mm.append(mean_mm)

    # Output per-marker result
    R, _ = cv2.Rodrigues(rvec)
    extrinsic = np.hstack((R, tvec.T))
    print(f"\n🧭 Marker ID {id_val}")
    print(f"Extrinsic Matrix [R|t]:\n{extrinsic}")
    print(f"📏 Mean Reprojection Error: {mean_px:.2f}px | {mean_mm:.2f}mm")

# Display total error
print(f"\n✅ Average Reprojection Error across all markers:")
print(f"   {np.mean(total_error_px):.2f} px | {np.mean(total_error_mm):.2f} mm")

# Show image
cv2.imshow("Reprojection: Green=real, Red=projected", vis_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
