import cv2
from ultralytics import YOLO
from transformers import pipeline

yolo = YOLO("yolov8m.pt")  # or yolov8n.pt
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def get_narration_overlay(frame):
    resized_frame = cv2.resize(frame, (640, 480))
    results = yolo.predict(resized_frame, verbose=False)
    names = yolo.names
    detected = set()

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            detected.add(names[cls])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, names[cls], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    labels = ", ".join(list(detected)[:5])
    if labels:
        prompt = f"The image contains the following objects: {labels}. Describe the scene in one short sentence."
        narration = generator(prompt, max_length=50, truncation=True)[0]['generated_text'].strip()
    else:
        narration = "The image contains no detectable objects."

    print("✅ Detected:", labels or "None")
    return frame, narration