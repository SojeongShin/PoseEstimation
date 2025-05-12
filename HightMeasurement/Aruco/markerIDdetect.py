import cv2
import numpy as np

# === Load camera parameters ===
intrin = np.load("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/cameraCali/intrinsic_calibration_result.npz")
K = intrin["K"]
dist = intrin["dist"]

# === Load image ===
img = cv2.imread("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/Aruco/arucoMat.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === Detect markers ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
params = cv2.aruco.DetectorParameters()
corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

# === Marker size (meters) ===
marker_length = 0.245

# === Pose estimation and drawing ===
if ids is not None:
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, K, dist)

    for i, id_val in enumerate(ids.flatten()):
        center = corners[i][0].mean(axis=0).astype(int)
        cv2.putText(img, f"ID:{id_val}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(img, tuple(center), 5, (255, 0, 0), -1)

        # Draw coordinate axes (red=X, green=Y, blue=Z)
        cv2.drawFrameAxes(img, K, dist, rvecs[i], tvecs[i], 0.1)  # 0.1m length

    cv2.imshow("Detected IDs with Axes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No markers detected.")
