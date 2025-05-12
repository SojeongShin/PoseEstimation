import cv2
import numpy as np
import glob

import os

# 폴더 생성
output_dir = "cali_frame"
os.makedirs(output_dir, exist_ok=True)

# 저장한 이미지 수 초기화
saved_count = 0



# 체스보드 설정
chessboard_size = (9, 6)  # 내부 코너 수 (행, 열)
square_size = 25  # mm 단위 (실제 체스보드 정사각형 한 칸의 길이)

# 3D 월드 좌표 생성 (z=0인 평면 위)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # 단위: mm

objpoints = []  # 3D point in real world
imgpoints = []  # 2D points in image plane

# 캘리브레이션 이미지 불러오기
images = glob.glob('HightMeasurement/cameraCali/valid_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        # 코너 정밀화
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        # 시각화
        img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)

        # 이미지 저장
        if saved_count < 10:
            save_path = os.path.join(output_dir, f"cali_{saved_count+1:02d}.jpg")
            cv2.imwrite(save_path, img)
            saved_count += 1

        # 시각화
        cv2.imshow('img', img)
        cv2.waitKey(100)


cv2.destroyAllWindows()

# 캘리브레이션 수행
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Intrinsic Matrix (K):\n", K)
print("Distortion Coefficients:\n", dist)

# 평균 한 칸의 픽셀 크기 계산
pixel_sizes = []

for i in range(len(imgpoints)):
    corners = imgpoints[i].reshape(-1, 2)

    # 가로 방향 기준 (한 행)
    row_dists = []
    for r in range(chessboard_size[1]):
        for c in range(chessboard_size[0] - 1):
            idx = r * chessboard_size[0] + c
            d = np.linalg.norm(corners[idx+1] - corners[idx])
            row_dists.append(d)

    # 세로 방향 기준 (한 열)
    col_dists = []
    for c in range(chessboard_size[0]):
        for r in range(chessboard_size[1] - 1):
            idx = r * chessboard_size[0] + c
            d = np.linalg.norm(corners[idx+chessboard_size[0]] - corners[idx])
            col_dists.append(d)

    mean_px = (np.mean(row_dists) + np.mean(col_dists)) / 2
    pixel_sizes.append(mean_px)

avg_pixels_per_square = np.mean(pixel_sizes)
print(f"✅ 평균 한 칸당 픽셀 수: {avg_pixels_per_square:.2f} px")

# 재투영 오차 계산
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)
print(f"✅ 평균 재투영 오차: {mean_error:.4f} px")

# mm 환산
square_size_mm = square_size  # 이미 mm 단위
mm_per_pixel = square_size_mm / avg_pixels_per_square
error_mm = mean_error * mm_per_pixel
print(f"✅ 재투영 오차 (실세계 기준): {error_mm:.3f} mm")

# 저장할 항목 구성
np.savez(
    "intrinsic_calibration_result.npz",
    K=K,                        # Intrinsic matrix
    dist=dist,                  # Distortion coefficients
    mean_reproj_error=mean_error,
    pixels_per_square=avg_pixels_per_square,
    square_size_mm=square_size,
    mm_per_pixel=mm_per_pixel,
    reprojection_error_mm=error_mm
)

print("📁 결과 저장 완료: intrinsic_calibration_forReport.npz")
