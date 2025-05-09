import cv2
import numpy as np
import os

# 체스보드 내부 코너 수 (예: 정사각형 10x7 → 내부 코너 9x6)
chessboard_size = (9, 6)

# 이미지 폴더 경로
image_dir = './valid_images'
image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.endswith('.jpg')]

# intrinsic 정보 불러오기
calib_data = np.load('intrinsic_calibration_result.npz')
camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['dist_coeffs']

# 3D 월드 좌표 기준점
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# 각 이미지에 대해 체스보드 코너 찾기
for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
    else:
        print(f"❌ 체스보드 검출 실패: {path}")

# 캘리브레이션 다시 수행 → rvecs, tvecs 얻기
_, _, _, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                            camera_matrix, dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

# 재투영 오차 계산
def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0
    total_points = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(imgpoints2)

    rms_error = np.sqrt(total_error / total_points)
    return rms_error

rms = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)
print(f"\n📏 최종 재투영 오차 (RMS): {rms:.4f} 픽셀")
