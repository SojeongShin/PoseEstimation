import cv2
import mediapipe as mp
import math
import time
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

PLM = mp_pose.PoseLandmark

# 오른쪽 측면 랜드마크
RIGHT_LANDMARKS = [
    PLM.RIGHT_SHOULDER,  # 12
    PLM.RIGHT_ELBOW,     # 14
    PLM.RIGHT_WRIST,     # 16
    PLM.RIGHT_INDEX,     # 20
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
    (PLM.RIGHT_WRIST, PLM.RIGHT_INDEX),
]


def compute_angle_knee(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y):
    """
    Knee(무릎)을 가운데로, (Hip->Knee)와 (Ankle->Knee) 두 벡터가 이루는 각도(도 단위)를 계산.
    - angle = 180이면 일직선(두 벡터가 정확히 반대 방향)
    - angle ~ 180이면 '거의 일직선'
    """
    ux, uy = hip_x - knee_x, hip_y - knee_y
    vx, vy = ankle_x - knee_x, ankle_y - knee_y

    dot = ux * vx + uy * vy
    mag_u = math.sqrt(ux**2 + uy**2)
    mag_v = math.sqrt(vx**2 + vy**2)

    if mag_u * mag_v == 0:
        return 0.0

    cos_theta = dot / (mag_u * mag_v)
    # 보정
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0

    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def draw_right_side_and_angle(image, landmarks, angle_offset=10, visibility_th=0.5):
    """
    1) 오른쪽 측면 랜드마크(점 + 연결선)를 파란색+흰색 테두리로 그린다.
    2) 각 점 옆에 (x, y) 픽셀 좌표를 표시한다.
    3) Hip-Knee-Ankle 사이각(= knee 각도)을 구해, 180도와 얼마나 가까운지 확인
       - angle >= 180 - angle_offset 이면 'Nearly Straight'이라고 표시
    4) foot index & hand index 의 y좌표 차이를 화면에 표시
    """

    h, w, _ = image.shape
    line_color = (255, 0, 0)   # 파란색 (BGR)
    white = (255, 255, 255)   # 흰색

    # 원하는 관절(hip, knee, ankle) 추출
    hip_lm = landmarks[PLM.RIGHT_HIP.value]
    knee_lm = landmarks[PLM.RIGHT_KNEE.value]
    ankle_lm = landmarks[PLM.RIGHT_ANKLE.value]

    # right_index 와 right_foot_index
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

        # (2) angle이 180 +- offset이면 'Nearly Straight'
        if (180 - angle_offset) <= angle <= (180 + angle_offset):
            angle_text += " (Nearly Straight)"

            # (3) y좌표 차이 (foot-index)
            li_y = right_index_lm.y * h
            lf_y = right_foot_index_lm.y * h
            dy = int(lf_y - li_y) * 0.26  # 픽셀 차이 -> 임의의 cm 환산
            dy_text = f"Y diff (foot - index): {dy:.1f}"

        pixel_dis = int(knee_y - hip_y)
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
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Lists to accumulate time, index y-values, foot y-values
    times = []
    right_index_ys = []
    right_foot_index_ys = []

    start_time = time.time()

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("카메라를 찾을 수 없습니다.")
                continue

            # 거울 모드 (좌우 반전)
            frame = cv2.flip(frame, 1)

            # Mediapipe Pose 처리
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # 오른쪽 관절 그리기 + Hip-Knee-Ankle 각도 표시
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 1) 호출해서 그림 그리기
                draw_right_side_and_angle(frame, landmarks, angle_offset=10, visibility_th=0.5)

                # 2) 오른손 검지(right_index), 오른발 끝(right_foot_index)의 Y값 기록
                #    visibility가 높은 경우만 기록해도 되고, 아래처럼 무조건 기록할 수도 있음.
                right_index_lm = landmarks[PLM.RIGHT_INDEX.value]
                right_foot_index_lm = landmarks[PLM.RIGHT_FOOT_INDEX.value]

                current_time = time.time() - start_time

                # 픽셀 단위로 변환
                frame_h = frame.shape[0]
                idx_y = right_index_lm.y * frame_h
                ft_y  = right_foot_index_lm.y * frame_h

                times.append(current_time)
                right_index_ys.append(idx_y)
                right_foot_index_ys.append(ft_y)

            cv2.imshow('Right side with angle', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC 키
                break

    cap.release()
    cv2.destroyAllWindows()

    # --------------------------------------------
    # After exiting the loop, save your two plots
    # --------------------------------------------
    # Plot 1: RIGHT_INDEX over time
    plt.figure(figsize=(8, 4))
    plt.plot(times, right_index_ys, color='blue', label='Right Index Y')
    plt.xlabel('Time (s)')
    plt.ylabel('Y position (pixels)')
    plt.title('Right Index Y vs Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plot/right_index_plot.png')  # Save as an image
    plt.close()

    # Plot 2: RIGHT_FOOT_INDEX over time
    plt.figure(figsize=(8, 4))
    plt.plot(times, right_foot_index_ys, color='red', label='Right Foot Index Y')
    plt.xlabel('Time (s)')
    plt.ylabel('Y position (pixels)')
    plt.title('Right Foot Index Y vs Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plot/right_foot_index_plot.png')  # Save as an image
    plt.close()


if __name__ == "__main__":
    main()
