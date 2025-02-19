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
# FaceAnalysis는 내부적으로 다양한 모델 선택 가능
app = FaceAnalysis(name='antelopev2')  
# 'antelopev2'에 ArcFace/MobileNet 계열이 포함됨. 
# 또는 app.prepare(ctx_id=0, det_size=(224,224)) 등 다양한 파라미터 세팅 가능.
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
    # FaceAnalysis app을 이용: app.get()에 이미지 전체를 넣으면,
    # 이미지 내의 얼굴들을 감지하고, face 객체를 반환
    # 다만 실시간 상황에서는 이미지 전체가 아닌 cropped face로 임베딩만 추출하는 예시를 사용할 수 있음
    # 여기서는 간단히 app.get_embeddings() 사용 예시 (버전에 따라 다름)
    
    face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)  # InsightFace는 BGR로 받을 수 있음
    # get embedding
    faces = app.get(face_img_bgr)  
    if len(faces) == 0:
        return None
    
    # 첫 번째 얼굴 임베딩만 사용
    emb = faces[0].embedding  # numpy array
    return emb

def compute_distance(emb1, emb2):
    """ 코사인 거리 or L2 거리 중 택 1 """
    # 코사인 유사도
    #   sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    #   distance = 1 - sim  (값이 작을수록 유사)
    
    # 여기선 L2 거리 사용 예시:
    return np.linalg.norm(emb1 - emb2)

# ---------------------------------------
# 5. 메인 루프 (카메라 스트림)
# ---------------------------------------
cap = cv2.VideoCapture(0)  # 웹캠 0번
with mp_face_detection.FaceDetection(
    model_selection=0,  # 0 or 1, 해상도/거리 범위에 따라 선택
    min_detection_confidence=0.5) as face_detector:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        if not tracking:
            # (A) 추적 중이 아니라면, MediaPipe로 얼굴 검출
            results = face_detector.process(frame_rgb)
            if results.detections:
                # 여러 얼굴이 잡히면 가장 큰 얼굴만 선택 (예시)
                max_area = 0
                chosen_detection = None
                for detection in results.detections:
                    # MediaPipe는 normalized 좌표를 제공
                    box = detection.location_data.relative_bounding_box
                    box_w = box.width * w
                    box_h = box.height * h
                    area = box_w * box_h
                    if area > max_area:
                        max_area = area
                        chosen_detection = detection
                
                if chosen_detection:
                    # 바운딩박스 계산
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
                    
                    face_crop = frame_rgb[y_min:y_min+box_h, x_min:x_min+box_w]
                    
                    # 임베딩 추출
                    emb = get_embedding(face_crop)
                    if emb is not None:
                        if registered_embedding is None:
                            # 아직 등록된 얼굴이 없으면, 이 얼굴을 등록
                            registered_embedding = emb
                            print("등록 완료!")
                            
                            # 추적기 초기화
                            tracker = cv2.TrackerKCF_create()
                            tracker.init(frame, (x_min, y_min, box_w, box_h))
                            tracking = True
                        else:
                            # 이미 등록된 사람과 같은지 비교
                            dist = compute_distance(registered_embedding, emb)
                            if dist < threshold:
                                print(f"동일인으로 판단 (distance={dist:.2f})")
                                # 동일인이면 추적 시작
                                tracker = cv2.TrackerKCF_create()
                                tracker.init(frame, (x_min, y_min, box_w, box_h))
                                tracking = True
                            else:
                                print(f"다른 사람으로 판단 (distance={dist:.2f})")
                                # 여기서는 '한 명만 추적'이 목표이므로 
                                # 다른 사람인 경우엔 무시 or 새 등록 등 로직을 정의하면 됨
        else:
            # (B) 추적 중인 경우
            success, bbox = tracker.update(frame)
            if success:
                # 추적 성공 → bbox 그리기
                x, y, w_box, h_box = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            else:
                # 추적 실패 → tracking 해제
                tracking = False
                tracker = None
                print("추적 실패. 다시 검출로 전환...")
        
        cv2.imshow("Face Tracking Demo", frame)
        
        # 종료 조건
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
