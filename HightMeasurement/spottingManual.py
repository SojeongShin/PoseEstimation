import cv2

image = cv2.imread("/home/sojeong/Documents/GitHub/PoseEstimation/HightMeasurement/cameraCali/rotated_frames/frame_1107.jpg")  # 촬영한 매트 이미지
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked point: ({x}, {y})")
        points.append([x, y])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", image)

cv2.imshow("Image", image)
cv2.setMouseCallback("Image", mouse_callback)

print("Click 3 points on the mat and press any key when done.")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Selected image points:", points)

# Click 3 points on the mat and press any key when done.
# Clicked point: (165, 837)
# Clicked point: (599, 834)
# Clicked point: (305, 963)
# Selected image points: [[165, 837], [599, 834], [305, 963]]