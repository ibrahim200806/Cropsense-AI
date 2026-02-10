import speech_recognition as sr
from gtts import gTTS
import tempfile
import os


def speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio)
    except Exception:
        return "Sorry, I could not understand."


def text_to_speech_bytes(text: str):
    """Return MP3 bytes for Streamlit audio player"""
    tts = gTTS(text)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp.name)

    with open(temp.name, "rb") as f:
        audio_bytes = f.read()

    os.remove(temp.name)
    return audio_bytes
    