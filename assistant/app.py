import cv2
from flask import Flask
from detection import detect_objects
from narration import narrate 

app = Flask(__name__)

@app.route("/")
def index():
    detected = detect_objects()
    narrate(detected)
    return f"Detected: {detected}"

if __name__ == "__main__":
    app.run(debug=True)
