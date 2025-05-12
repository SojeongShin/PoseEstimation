import cv2
import numpy as np
import os
import shutil

# 체스보드 내부 코너 수 (정사각형 10x7이면 → 내부 코너 9x6)
chessboard_size = (9, 6)

# 디렉토리 설정
input_dir = './rotated_frames'
output_dir = './valid_images'
os.makedirs(output_dir, exist_ok=True)

# 3D 기준점 (z=0 평면 상)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 결과 저장용 리스트
objpoints = []  # 3D 실제 좌표
imgpoints = []  # 2D 이미지 좌표

valid_images = []

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith('.jpg'):
        continue
    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        valid_images.append(fname)

        # 유효 이미지 저장
        shutil.copy(img_path, os.path.join(output_dir, fname))
        print(f"✔️ 검출 성공: {fname}")
    else:
        print(f"❌ 검출 실패: {fname}")

print("\n✅ 최종 유효 이미지 수:", len(valid_images))
