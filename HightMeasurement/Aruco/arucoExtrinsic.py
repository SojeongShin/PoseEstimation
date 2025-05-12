import cv2
import numpy as np

# === Load intrinsic parameters ===
intrinsic_data = np.load('HightMeasurement/cameraCali/intrinsic_calibration_result.npz')
K = intrinsic_data['K']              # Intrinsic matrix (3x3)
dist = intrinsic_data['dist']       # Distortion coefficients (1x5 or 1x8)

# === Load image ===
image = cv2.imread('arucoMat.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === Detect ArUco markers ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters_create()
corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

if ids is None or len(ids) < 4:
    raise ValueError("❌ Less than 4 markers detected.")

# === Define world positions of markers ===
# Example: 4 markers forming a 1m x 1m square (you can change units or spacing)
# You must replace these with real-world measurements in meters
id_to_world = {
    0: [0.0, 0.0, 0.0],
    1: [1.0, 0.0, 0.0],
    2: [1.0, 1.0, 0.0],
    3: [0.0, 1.0, 0.0]
}

# === Build matching 2D-3D points ===
objpoints = []
imgpoints = []

for i, id_val in enumerate(ids.flatten()):
    if id_val in id_to_world:
        center = corners[i][0].mean(axis=0)  # Average of 4 corner points
        imgpoints.append(center)
        objpoints.append(id_to_world[id_val])

objpoints = np.array(objpoints, dtype=np.float32)
imgpoints = np.array(imgpoints, dtype=np.float32)

if len(objpoints) < 4:
    raise ValueError("❌ Not enough known IDs matched for pose estimation.")

# === Solve for pose ===
success, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, K, dist)
R, _ = cv2.Rodrigues(rvec)
extrinsic = np.hstack((R, tvec))  # 3x4 matrix: [R | t]

# === Output ===
print("✅ Extrinsic Matrix [R | t]:\n", extrinsic)
