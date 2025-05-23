import librosa
import parselmouth
import numpy as np
from pydub import AudioSegment

def preprocess_audio(audio_path):
    """Функция F1: предобработка аудиозаписи"""
    # Конвертация в нужный формат при необходимости
    audio = AudioSegment.from_wav(audio_path)
    processed_path = "processed_audio.wav"
    audio.export(processed_path, format="wav")
    
    # Загрузка аудио для анализа
    y, sr = librosa.load(processed_path)
    sound = parselmouth.Sound(processed_path)
    
    return y, sr, sound