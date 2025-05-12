import cv2
import numpy as np

# Load image
img = cv2.imread("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/Aruco/arucoMat.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
params = cv2.aruco.DetectorParameters()
corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

# Draw and display IDs at marker centers
if ids is not None:
    for i, id_val in enumerate(ids.flatten()):
        # Find marker center
        c = corners[i][0].mean(axis=0).astype(int)
        cv2.putText(img, f"ID:{id_val}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.circle(img, tuple(c), 5, (255,0,0), -1)

    cv2.imshow("Detected IDs", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No markers detected.")
