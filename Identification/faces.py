import cv2
import mediapipe as mp
import numpy as np
import pickle
import tflite_runtime.interpreter as tflite

# 1. Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# 2. Initialize TFLite Face Embedding Model
#    Replace "face_embedding_model.tflite" with your actual model file
interpreter = tflite.Interpreter(model_path="Identification/friends.jpg")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Helper to get embeddings
def get_face_embedding(face_bgr):
    # Convert to RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    # Preprocess as required by your TFLite model
    # e.g., resize to 112x112, 160x160, or whatever your model expects
    face_resized = cv2.resize(face_rgb, (160, 160))
    face_resized = np.expand_dims(face_resized, axis=0).astype(np.float32)

    # Model might require normalization, e.g. face_resized /= 255.0

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], face_resized)
    interpreter.invoke()

    embedding = interpreter.get_tensor(output_details[0]['index'])
    # embedding might be (1,128), (1,512), etc. Flatten if needed:
    embedding = embedding[0]
    return embedding

# Load an image
img_bgr = cv2.imread("Identification/friends.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:

    # Detect faces
    results = face_detection.process(img_rgb)
    if not results.detections:
        print("No faces detected.")
    else:
        for i, detection in enumerate(results.detections):
            # Convert relative coords to absolute
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img_bgr.shape
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            x_max = x_min + width
            y_max = y_min + height
            
            # Crop the face
            cropped_face = img_bgr[y_min:y_max, x_min:x_max]
            out_path = f"stored-faces/{i}.jpg"
            cv2.imwrite(out_path, cropped_face)

            # Get face embedding
            embedding = get_face_embedding(cropped_face)

            # Save embedding to pickle
            with open(f"embedding_{i}.pkl", "wb") as f:
                pickle.dump(embedding, f)

            print(f"Saved embedding_{i}.pkl")


import numpy as np

with open("embedding_0.pkl", "rb") as f:
    reference_embedding = pickle.load(f)

new_embedding = get_face_embedding(cropped_face)  # from new image

distance = np.linalg.norm(new_embedding - reference_embedding)
if distance < 0.9:  # threshold depends on your model
    print("Same person!")
else:
    print("Different person!")
