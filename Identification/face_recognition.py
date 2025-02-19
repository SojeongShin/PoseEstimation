import cv2
import mediapipe as mp
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time

# 1. MediaPipe settings
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 2. InsightFace: MobileFaceNet or ArcFace
app = FaceAnalysis(
    allowed_modules=['detection', 'recognition'], 
    providers=['CPUExecutionProvider']
)  
app.prepare(ctx_id=0, det_size=(224, 224))

# 3. Registration / detection thresholds
SIM_THRESHOLD = 0.5  # If cos_sim > 0.5, we say "Same Person"

# 4. Global variables
registered_embedding = None  # We'll set this once after the 3-second capture
countdown_duration = 10      # seconds for initial "Get ready" countdown
capture_duration = 3         # seconds to capture/choose the best face after countdown
countdown_done = False       # has the 10s countdown finished?
capturing = False            # are we in the 3s capturing window?
countdown_start_time = time.time()
capture_start_time = None

# We'll store the "best face" (largest bounding box area) and its embedding
best_face_area = 0
best_face_emb = None

def get_embedding(face_img_rgb):
    """
    Convert face region to BGR, run through InsightFace, return embedding.
    """
    face_bgr = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2BGR)
    faces = app.get(face_bgr)
    if len(faces) == 0:
        return None
    return faces[0].embedding

def compute_cosine_similarity(emb1, emb2):
    """
    Cosine similarity (closer to 1 => more similar).
    If you want a 'distance', you could do 1 - cosine_similarity.
    """
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1_norm, emb2_norm)

# 5. Main loop (camera stream)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 768)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
cap.set(cv2.CAP_PROP_FPS, 27)

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
) as face_detector:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate/flip if needed
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        now = time.time()

        # -------------------------------
        # A) Countdown Phase
        # -------------------------------
        if not countdown_done and registered_embedding is None:
            elapsed = now - countdown_start_time
            remain = int(countdown_duration - elapsed)
            if remain > 0:
                # Show countdown text
                text = f"Get ready! Registration in {remain} s"
                font_scale = 1.2
                thickness = 3
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                x_pos = (w - text_w) // 2
                y_pos = (h + text_h) // 2
                cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
            else:
                # Countdown finished -> Start capturing
                countdown_done = True
                capturing = True
                capture_start_time = time.time()
                print("[INFO] Countdown finished. Starting 3s face capture phase...")

        # -------------------------------
        # B) Face Capture Phase (3s)
        # -------------------------------
        elif capturing and registered_embedding is None:
            capture_elapsed = now - capture_start_time
            remain_cap = int(capture_duration - capture_elapsed)
            if remain_cap > 0:
                # Show message "Capturing face X sec left"
                text = f"Capturing face... {remain_cap} s remaining"
                font_scale = 1.0
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                x_pos = (w - text_w) // 2
                y_pos = (h + text_h) // 2
                cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
                
                # Detect face(s) each frame, keep track of largest bounding box + embedding
                results = face_detector.process(frame_rgb)
                if results.detections:
                    for detection in results.detections:
                        box = detection.location_data.relative_bounding_box
                        x_min = int(box.xmin * w)
                        y_min = int(box.ymin * h)
                        box_w = int(box.width * w)
                        box_h = int(box.height * h)

                        # Safe clamp
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        box_w = min(w - x_min, box_w)
                        box_h = min(h - y_min, box_h)

                        area = box_w * box_h
                        # If this face is largest so far, store it
                        if area > best_face_area:
                            face_crop = frame_rgb[y_min:y_min+box_h, x_min:x_min+box_w]
                            emb = get_embedding(face_crop)
                            if emb is not None:
                                best_face_area = area
                                best_face_emb = emb

                        # Draw bounding box
                        cv2.rectangle(frame, (x_min, y_min),
                                      (x_min + box_w, y_min + box_h),
                                      (0, 255, 255), 2)
                        cv2.putText(frame, "Capturing Face", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            else:
                # 3s capture done -> finalize registration with best_face_emb
                capturing = False
                if best_face_emb is not None:
                    registered_embedding = best_face_emb
                    print("[INFO] Face registration complete!")
                else:
                    print("[WARN] No valid face found in 3s capture. Registration still None.")

        # -------------------------------
        # C) Normal Operation (Compare)
        # -------------------------------
        else:
            # If we have a registered embedding, compare new faces
            if registered_embedding is not None:
                # Detect face(s)
                results = face_detector.process(frame_rgb)
                if results.detections:
                    for detection in results.detections:
                        box = detection.location_data.relative_bounding_box
                        x_min = int(box.xmin * w)
                        y_min = int(box.ymin * h)
                        box_w = int(box.width * w)
                        box_h = int(box.height * h)

                        # Safe clamp
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        box_w = min(w - x_min, box_w)
                        box_h = min(h - y_min, box_h)

                        face_crop = frame_rgb[y_min:y_min+box_h, x_min:x_min+box_w]
                        emb = get_embedding(face_crop)
                        if emb is not None:
                            sim = compute_cosine_similarity(registered_embedding, emb)
                            if sim > SIM_THRESHOLD:
                                # Same person => highlight in green
                                color = (0, 255, 0)
                                label = f"Same Person (sim={sim:.2f})"
                            else:
                                # Different => highlight in red
                                color = (0, 0, 255)
                                label = f"Different (sim={sim:.2f})"
                            
                            cv2.rectangle(frame, (x_min, y_min),
                                          (x_min+box_w, y_min+box_h),
                                          color, 2)
                            cv2.putText(frame, label, (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            else:
                # Still no valid registration, just show a message
                text = "Waiting for registration..."
                font_scale = 1.0
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                x_pos = (w - text_w) // 2
                y_pos = (h + text_h) // 2
                cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        cv2.imshow("Face Capture & Recognition Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
