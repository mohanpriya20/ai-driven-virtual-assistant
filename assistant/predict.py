import tensorflow as tf
import cv2
import numpy as np

# Load .h5 model (ensure this file exists!)
model = tf.keras.models.load_model("assistant/model/my_model.h5")

def predict_frame():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return "Nothing detected"

    image = cv2.resize(frame, (224, 224)) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]
    predicted_class = np.argmax(prediction)
    
    # Update class map if needed
    class_map = ["man reading a book", "person with laptop", "empty room"]
    label = class_map[predicted_class]
    return label
