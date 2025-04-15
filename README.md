# 🧠 AI-Powered Virtual Assistant

This project is an intelligent assistant that combines computer vision and speech synthesis to detect and narrate real-world objects via webcam, logging them in a local file.

## 💡 Features
- Real-time object detection with TensorFlow + OpenCV
- Text-to-Speech narration using Hugging Face Transformers or Google TTS
- Logs recognized objects with timestamps
- Docker-ready deployment

## 🚀 Getting Started

1. Clone the repo:
```bash
git clone https://github.com/mohanpriya20/ai-virtual-assistant.git
cd ai-virtual-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the assistant:
```bash
python assistant/app.py
```

## 🐳 Docker
To run in Docker:
```bash
docker build -t virtual-assistant .
docker run -p 5000:5000 virtual-assistant
```
