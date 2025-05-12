import cv2
import os

# 원하는 최대 해상도 (카메라가 지원하는 해상도에 따라 조정)
TARGET_WIDTH = 1024
TARGET_HEIGHT = 768

save_dir = "./rotated_frames"
os.makedirs(save_dir, exist_ok=True)

# 웹캠 열기q
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

# 실제 적용된 해상도 확인
ret, test_frame = cap.read()
if not ret:
    print("❌ 카메라에서 프레임을 읽을 수 없습니다.")
    exit()

print(f"📐 실제 해상도: {test_frame.shape[1]}x{test_frame.shape[0]} (width x height)")

frame_count = 0
print("🎥 캡처 시작! q 키를 누르면 종료합니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    # 시계 방향 90도 회전 → 결과 해상도: (768, 1024)
    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # 저장
    filename = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(filename, rotated)
    print(f"📸 저장됨: {filename}")
    frame_count += 1

    # 프리뷰 (선택)
    cv2.imshow("Rotated Frame", rotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 사용자가 중단함")
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ 총 {frame_count}장 저장 완료.")
