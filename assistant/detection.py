import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the pre-trained SSD model (COCO classes)
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
labels = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect_objects():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "nothing"

    # Resize + normalize
    img = cv2.resize(frame, (320, 320))
    img = np.expand_dims(img / 255.0, axis=0).astype(np.float32)

    # Run model
    results = detector(img)
    result = {key: val.numpy() for key, val in results.items()}

    classes = result["detection_classes"][0]
    scores = result["detection_scores"][0]

    print("Detection scores:", scores[:5])  # 👈 DEBUG LINE

    detected_objects = set()
    for i in range(len(scores)):
        if scores[i] > 0.2:
            class_id = int(classes[i])
            detected_objects.add(labels[class_id])

    return ", ".join(detected_objects) if detected_objects else "nothing"
