import cv2
import numpy as np

def wp(x, w): return int(w * x)
def hp(y, h): return int(h * y)

def draw_overlay(frame):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    blue_color = (255, 0, 0)
    green_color = (0, 255, 0)
    alpha = 0.5

    # === Horizontal scaling and centering setup ===
    scale = 2.6
    original_center = 0.76  # Estimated center of original shape
    target_center = 0.5     # Center of the frame

    def transform_x(x):
        return (x - original_center) * scale + target_center

    lines = [
        [(0.485, 1.02), (0.5975, 0.77)],
        [(0.5975, 0.77), (0.89875, 0.77)],
        [(0.89875, 0.77), (1.0114, 1.02)],
        [(0.6475, 0.77), (0.635, 0.805)],
        [(0.635, 0.805), (0.58, 0.805)],
        [(0.84875, 0.77), (0.86125, 0.805)],
        [(0.86125, 0.805), (0.91625, 0.805)],
        [(0.675, 0.88), (0.670, 0.95)],
        [(0.670, 0.95), (0.830, 0.95)],
        [(0.830, 0.95), (0.825, 0.885)],
        [(0.825, 0.885), (0.6715, 0.885)],
        [(0.6745, 0.884), (0.68, 0.850)],
        [(0.68, 0.850), (0.82, 0.850)],
        [(0.82, 0.850), (0.825, 0.881)],
    ]

    for (x1, y1), (x2, y2) in lines:
        pt1 = (wp(transform_x(x1), w), hp(y1, h))
        pt2 = (wp(transform_x(x2), w), hp(y2, h))
        cv2.line(overlay, pt1, pt2, blue_color, thickness=5)

    # Green line (horizontally stretched and centered)
    gx1, gx2 = transform_x(0.68), transform_x(0.82)
    pt1 = (wp(gx1, w), hp(0.850, h))
    pt2 = (wp(gx2, w), hp(0.850, h))
    cv2.line(overlay, pt1, pt2, green_color, thickness=5)

    # Blend with transparency
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


# === Camera Setup ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
cap.set(cv2.CAP_PROP_FPS, 27)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate frame 90 degrees clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    draw_overlay(frame)

    cv2.imshow('Kiosk Overlay - Centered + Stretched', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
