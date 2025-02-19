import cv2
import mediapipe as mp
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# ---------------------------------------
# 1. MediaPipe 설정(BlazeFace 기반)
# ---------------------------------------
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------
# 2. InsightFace: MobileFaceNet 모델 로드
# ---------------------------------------
app = FaceAnalysis(
    allowed_modules=['detection', 'recognition'], 
    providers=['CPUExecutionProvider']
)  
app.prepare(ctx_id=0, det_size=(224,224))

# ---------------------------------------
# 3. OpenCV Tracker(KCF) 초기화
# ---------------------------------------
tracker = None
tracking = False

# ---------------------------------------
# 4. 등록된 사용자 얼굴 임베딩 (None이면 아직 미등록)
# ---------------------------------------
registered_embedding = None
threshold = 1.0  # 예시값, 실제로는 모델·거리 스케일에 따라 조정

def get_embedding(face_img):
    """
    주어진 얼굴 이미지(RGB 3D numpy array)에 대해
    InsightFace의 MobileFaceNet/ArcFace 등으로 임베딩을 구해 반환.
    """
    face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    faces = app.get(face_img_bgr)  
    if len(faces) == 0:
        return None
    
    emb = faces[0].embedding
    return emb

def compute_distance(emb1, emb2):
    """ 코사인 거리 또는 L2 거리 """
    return np.linalg.norm(emb1 - emb2)


# ---------------------------------------
# 5. 메인 루프 (카메라 스트림)
# ---------------------------------------
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('video/squat.webm')


# Adjust resolution and FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 768)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
cap.set(cv2.CAP_PROP_FPS, 27)

with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6) as face_detector:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate and flip (if needed)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        if not tracking:
            # (A) 추적 중이 아닐 때 → MediaPipe로 얼굴 검출
            results = face_detector.process(frame_rgb)
            if results.detections:
                # 여러 얼굴이 잡히면 가장 큰 얼굴만 선택 (예시)
                max_area = 0
                chosen_detection = None
                for detection in results.detections:

                    # Check detection confidence
                    # NEW: Additional filter (e.g. > 0.7)
                    if detection.score[0] < 0.5:
                        continue


                    box = detection.location_data.relative_bounding_box
                    box_w = box.width * w
                    box_h = box.height * h
                    area = box_w * box_h
                    if area > max_area:
                        max_area = area
                        chosen_detection = detection
                
                if chosen_detection:
                    rb = chosen_detection.location_data.relative_bounding_box
                    x_min = int(rb.xmin * w)
                    y_min = int(rb.ymin * h)
                    box_w = int(rb.width * w)
                    box_h = int(rb.height * h)

                    # 잘려나가지 않도록 안전 처리
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    box_w = min(w - x_min, box_w)
                    box_h = min(h - y_min, box_h)

                    # ----------------------------------------------------
                    # ADDED: Draw bounding box for face detection
                    # ----------------------------------------------------
                    cv2.rectangle(frame, (x_min, y_min), (x_min+box_w, y_min+box_h),
                                  (0, 255, 255), 2)
                    cv2.putText(frame, "Detected Face", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    face_crop = frame_rgb[y_min:y_min+box_h, x_min:x_min+box_w]
                    
                    # 임베딩 추출
                    emb = get_embedding(face_crop)
                    if emb is not None:
                        if registered_embedding is None:
                            # 아직 등록된 얼굴이 없으면, 등록
                            registered_embedding = emb
                            print("등록 완료!")
                            
                            # 추적기 초기화
                            tracker = cv2.TrackerKCF_create()
                            tracker.init(frame, (x_min, y_min, box_w, box_h))
                            tracking = True

                        else:
                            # 이미 등록된 사람과 비교
                            dist = compute_distance(registered_embedding, emb)
                            if dist < threshold:
                                print(f"동일인으로 판단 (distance={dist:.2f})")
                                # 추적 시작
                                tracker = cv2.TrackerKCF_create()
                                tracker.init(frame, (x_min, y_min, box_w, box_h))
                                tracking = True
                            else:
                                print(f"다른 사람으로 판단 (distance={dist:.2f})")
                                # 한 명만 추적 → 무시 or 새 등록 등 로직 가능
        else:
            # (B) 추적 중인 경우
            success, bbox = tracker.update(frame)
            if success:
                x, y, w_box, h_box = map(int, bbox)
                
                # ----------------------------------------------------
                # ADDED: Draw bounding box for face tracking
                # ----------------------------------------------------
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box),
                              (0, 255, 0), 2)
                cv2.putText(frame, "Tracking Face", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # 추적 실패 → tracking 해제
                tracking = False
                tracker = None
                print("추적 실패. 다시 검출로 전환...")
        
        cv2.imshow("Face Tracking Demo", frame)
        
        # 종료 조건 (ESC)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
