from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image
import torch
import numpy as np
import cv2

# === Load weights and model with correct syntax ===
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights).eval()

# === Define preprocessing ===
preprocess = weights.transforms()

# === Load and preprocess image ===
img_path = "/Users/sojeongshin/Documents/GitHub/PoseEstimation/HightMeasurement/ref/sj/back.jpg"
img_pil = Image.open(img_path).convert("RGB")
input_tensor = preprocess(img_pil).unsqueeze(0)

# === Run inference ===
with torch.no_grad():
    output = model(input_tensor)["out"][0]
mask = output.argmax(0).byte().cpu().numpy()

# === Get person class mask (class 15) ===
person_mask = (mask == 15).astype(np.uint8) * 255

# === OpenCV visualization with overlay ===
img_cv = cv2.imread(img_path)
mask_resized = cv2.resize(person_mask, (img_cv.shape[1], img_cv.shape[0]))

# Create overlay image
overlay = img_cv.copy()
color = (255, 0, 0)  # Blue (BGR)
alpha = 0.5  # Transparency

# Apply color to person mask only
colored_area = mask_resized == 255
overlay[colored_area] = (
    alpha * np.array(color) + (1 - alpha) * overlay[colored_area]
).astype(np.uint8)

# Show the result
cv2.imshow("Person Segmentation (Overlay)", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# (Optional) Save the image
cv2.imwrite("segmentation_overlay.jpg", overlay)
