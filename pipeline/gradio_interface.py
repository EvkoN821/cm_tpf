import gradio as gr
from random import randint
from pydub import AudioSegment
from transcription_processing import transcribe_and_process
from feature_extraction import extract_features
from summary_generation import generate_summary

def process_audio(file):
    """Обработка аудиофайла через интерфейс Gradio"""
    # Конвертация аудио
    audio = AudioSegment.from_wav(file)
    temp_path = f"temp_audio_{randint(1, 10**8)}.wav"
    audio.export(temp_path, format="wav")
    
    # Выполнение пайплайна
    sentences, word_timestamps, tfidf_scores, textrank_scores = transcribe_and_process(temp_path)
    features_df = extract_features(temp_path, sentences, word_timestamps, tfidf_scores, textrank_scores)
    summary, _ = generate_summary(features_df)
    
    return "".join(sentences), summary

def create_interface():
    """Создание Gradio интерфейса"""
    demo = gr.Interface(
        process_audio,
        gr.Audio(type="filepath"),
        [
            gr.Textbox(value="", label="Исходный текст"),
            gr.Textbox(value="", label="Сокращенный текст")
        ]
    )
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()