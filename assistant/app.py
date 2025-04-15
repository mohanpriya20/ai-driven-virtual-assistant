from flask import Flask, render_template, Response, jsonify
from assistant.utils import get_narration_overlay
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_narration = "Starting narration..."

def generate_frames():
    global current_narration
    while True:
        success, frame = camera.read()
        if not success or frame is None or frame.shape[0] == 0:
            print("❌ Invalid frame.")
            continue

        frame, narration = get_narration_overlay(frame)
        current_narration = narration

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("❌ Frame encoding failed.")
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/narration')
def narration():
    return jsonify({'narration': current_narration})

if __name__ == '__main__':
    app.run(debug=True)
