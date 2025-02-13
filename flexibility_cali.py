import cv2
import mediapipe as mp
import math
import numpy as np

# ======== YOUR CAMERA CALIBRATION RESULTS =========
camera_matrix = np.array([
    [801.36977545,   0.        , 508.42593294],
    [  0.        , 801.32338727, 375.9853481 ],
    [  0.        ,   0.        ,   1.        ]
])

dist_coeffs = np.array([
    [-0.00988595,  0.35595046,  0.00719413, -0.00249943, -1.15314737]
])
# ===================================================

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# PoseLandmark를 간단히 쓰기 위한 별칭
PLM = mp_pose.PoseLandmark

# 오른쪽 측면 랜드마크
RIGHT_LANDMARKS = [
    PLM.RIGHT_SHOULDER,  # 12
    PLM.RIGHT_ELBOW,     # 14
    PLM.RIGHT_WRIST,     # 16
    # PLM.RIGHT_PINKY,     # 18
    PLM.RIGHT_INDEX,     # 20
    # PLM.RIGHT_THUMB,     # 22
    PLM.RIGHT_HIP,       # 24
    PLM.RIGHT_KNEE,      # 26
    PLM.RIGHT_ANKLE,     # 28
    PLM.RIGHT_HEEL,      # 30
    PLM.RIGHT_FOOT_INDEX # 32
]

# 오른쪽 측면 연결 관계
RIGHT_CONNECTIONS = [
    (PLM.RIGHT_SHOULDER, PLM.RIGHT_ELBOW),
    (PLM.RIGHT_ELBOW, PLM.RIGHT_WRIST),
    (PLM.RIGHT_SHOULDER, PLM.RIGHT_HIP),
    (PLM.RIGHT_HIP, PLM.RIGHT_KNEE),
    (PLM.RIGHT_KNEE, PLM.RIGHT_ANKLE),
    (PLM.RIGHT_ANKLE, PLM.RIGHT_HEEL),
    (PLM.RIGHT_HEEL, PLM.RIGHT_FOOT_INDEX),
    # (PLM.RIGHT_WRIST, PLM.RIGHT_PINKY),
    (PLM.RIGHT_WRIST, PLM.RIGHT_INDEX),
    # (PLM.RIGHT_WRIST, PLM.RIGHT_THUMB),
]


def compute_angle_knee(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y):
    """
    Knee(무릎)을 가운데로, (Hip->Knee)와 (Ankle->Knee) 두 벡터가 이루는 각도(도 단위)를 계산.
    - angle = 180이면 일직선(두 벡터가 정확히 반대 방향)
    - angle ~ 180이면 '거의 일직선'
    """
    # 벡터 u = (Hip - Knee), v = (Ankle - Knee)
    ux, uy = hip_x - knee_x, hip_y - knee_y
    vx, vy = ankle_x - knee_x, ankle_y - knee_y

    # 두 벡터의 크기와 내적
    dot = ux * vx + uy * vy
    mag_u = math.sqrt(ux**2 + uy**2)
    mag_v = math.sqrt(vx**2 + vy**2)

    # 혹시 0으로 나누는 경우 방지
    if mag_u * mag_v == 0:
        return 0.0

    # cos(theta) = (u·v) / (|u|*|v|)
    cos_theta = dot / (mag_u * mag_v)
    # float 오차로 인해 범위를 벗어나면 보정
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0

    # angle (라디안) -> 도(degrees)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def draw_right_side_and_angle(image, landmarks, angle_offset=10, visibility_th=0.5):
    """
    1) 오른쪽 측면 랜드마크(점 + 연결선)를 파란색+흰색 테두리로 그린다.
    2) 각 점 옆에 (x, y) 픽셀 좌표를 표시한다.
    3) Hip-Knee-Ankle 사이각(= knee 각도)을 구해, 180도와 얼마나 가까운지 확인
       - angle >= 180 - angle_offset 이면 '거의 일직선'이라고 표시
    4) Hip, Knee, Ankle 의 y좌표 차이를 화면에 표시
    """

    h, w, _ = image.shape
    line_color = (255, 0, 0)   # 파란색 (BGR)
    white = (255, 255, 255)   # 흰색

    # 원하는 관절(hip, knee, ankle) 추출
    hip_lm = landmarks[PLM.RIGHT_HIP.value]
    knee_lm = landmarks[PLM.RIGHT_KNEE.value]
    ankle_lm = landmarks[PLM.RIGHT_ANKLE.value]

    # right_foor_index와 right_index 사이 거리
    right_index_lm = landmarks[PLM.RIGHT_INDEX.value]
    right_foot_index_lm = landmarks[PLM.RIGHT_FOOT_INDEX.value]

    angle_text = ""
    dy_text = ""
    pixel_text = ""

    # 세 관절 모두 visibility가 충분하면 각도 계산
    if (hip_lm.visibility > visibility_th and
        knee_lm.visibility > visibility_th and
        ankle_lm.visibility > visibility_th):

        hip_x, hip_y = hip_lm.x * w, hip_lm.y * h
        knee_x, knee_y = knee_lm.x * w, knee_lm.y * h
        ankle_x, ankle_y = ankle_lm.x * w, ankle_lm.y * h

        # (1) 무릎 기준 각도 계산
        angle = compute_angle_knee(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y)
        angle_text = f"Angle(Hip-Knee-Ankle): {angle:.1f} deg"

        # (2) angle이 180 - offset 이상이면 '거의 일직선'
        if angle >= (180 - angle_offset):
            angle_text += " (Nearly Straight)"

            # (3) y좌표 차이 (foot-index)
            li_y = right_index_lm.y * h
            lf_y = right_foot_index_lm.y * h
            # 픽셀 단위 차이를 0.26cm로 환산(예시)
            dy = int(lf_y - li_y) * 0.26  
            dy_text = f"Y diff (foot - index): {dy:.1f} cm"

        pixel_dis = (knee_y - hip_y)
        pixel_text = f"pixel: {pixel_dis:.1f}"

    # (A) 오른쪽 측면 랜드마크 & 좌표 표시
    for lm_id in RIGHT_LANDMARKS:
        lm = landmarks[lm_id.value]
        if lm.visibility < visibility_th:
            continue

        cx, cy = int(lm.x * w), int(lm.y * h)

        # 1) 흰색 테두리 원
        cv2.circle(image, (cx, cy), 12, white, 6)
        # 2) 파란색 꽉 채운 원
        cv2.circle(image, (cx, cy), 7, line_color, -1)

        # 3) (cx, cy) 텍스트 표시
        text_str = f"({cx}, {cy})"
        cv2.putText(
            image, text_str,
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # 폰트 크기
            (255, 255, 255),  # 흰색 글씨
            1,
            cv2.LINE_AA
        )

    # (B) 오른쪽 측면 연결선
    for conn in RIGHT_CONNECTIONS:
        lm1 = landmarks[conn[0].value]
        lm2 = landmarks[conn[1].value]

        if lm1.visibility < visibility_th or lm2.visibility < visibility_th:
            continue

        x1, y1 = int(lm1.x * w), int(lm1.y * h)
        x2, y2 = int(lm2.x * w), int(lm2.y * h)
        cv2.line(image, (x1, y1), (x2, y2), line_color, 5)

    # (C) 화면 상단에 angle, y-diff 표시
    if angle_text:
        cv2.putText(
            image, angle_text, (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (255, 255, 255), 2, cv2.LINE_AA
        )
    if dy_text:
        cv2.putText(
            image, dy_text, (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 2, cv2.LINE_AA
        )
    if pixel_text:
        cv2.putText(
            image, pixel_text, (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 2, cv2.LINE_AA
        )


def main():
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    cap.set(cv2.CAP_PROP_FPS, 27)

    # Grab one frame just to set up undistortion maps, etc.
    ret, test_frame = cap.read()
    if not ret:
        print("Failed to read from camera. Exiting.")
        return

    h, w = test_frame.shape[:2]

    # Optional: get a new (optimized) camera matrix for this resolution
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("카메라를 찾을 수 없습니다.")
                break

            # --- STEP 1: Undistort the frame using calibration data ---
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)

            # (Optional) Crop the valid ROI if you want:
            # x, y, w_roi, h_roi = roi
            # undistorted = undistorted[y:y+h_roi, x:x+w_roi]

            # 거울 모드 (좌우 반전)
            undistorted = cv2.flip(undistorted, 1)

            # --- STEP 2: Mediapipe Pose 처리 (on undistorted frame) ---
            frame_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # --- STEP 3: 오른쪽 관절 그리기 + Hip-Knee-Ankle 각도 표시 ---
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                draw_right_side_and_angle(undistorted, landmarks, angle_offset=10, visibility_th=0.5)

            # Show result
            cv2.imshow('Right side with angle (Undistorted)', undistorted)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
