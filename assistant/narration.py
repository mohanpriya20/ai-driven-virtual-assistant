from gtts import gTTS
import os

def narrate(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("start output.mp3" if os.name == "nt" else "afplay output.mp3")
