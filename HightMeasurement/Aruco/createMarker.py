import cv2
import cv2.aruco as aruco
import os
import numpy as np

# Dictionary 선택
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

# 저장 경로
os.makedirs("aruco_markers", exist_ok=True)

# 마커 크기 (픽셀)
marker_size = 700

# ID 0~3번 마커 생성
for marker_id in range(4):
    marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
    aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_img, 1)
    cv2.imwrite(f"aruco_markers/aruco5x5_100_id{marker_id}.png", marker_img)

print("✅ 마커 4개 생성 완료 (generateImageMarker 사용)")
