from ultralytics import YOLO

# Choose one of the larger models
YOLO("yolov8m.pt")  # Medium model (more accurate than yolov8n)
# YOLO("yolov8x.pt")  # Extra-large model (most accurate, but heavier)

print("✅ YOLOv8 model downloaded.")
