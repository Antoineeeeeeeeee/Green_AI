from pathlib import Path
from ultralytics import YOLO
import cv2

# Directory and file paths
PHOTOS_DIR = 'photos'
photo_path = str(Path(PHOTOS_DIR) / 'GettyImages-1287574718-scaled.jpg')
path_out = str(Path(PHOTOS_DIR) / 'n3.jpg')

# Load image
img = cv2.imread(photo_path)


# Load YOLO model
model_path = 'weights/best.pt'
model = YOLO(model_path)

threshold = 0.5

# Run YOLO on the image
results = model(img)[0]

# Draw detections
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        # Draw rectangle
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

        # Add label with confidence
        label = f"{results.names[int(class_id)].upper()} {score:.2f}"
        cv2.putText(
            img,
            label,
            (int(x1), int(y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )


# Save output image
cv2.imwrite(path_out, img)
print(f"Saved: {path_out}")
