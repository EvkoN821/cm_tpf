import numpy as np
import pandas as pd
import librosa
import parselmouth
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import torch
import whisper
from random import randint
import gradio as gr
from pydub import AudioSegment



nltk.download('punkt_tab')
DEFAULT_WEIGHTS = {
    'tfidf': 0.3,
    'textrank': 0.3,
    'position': 0.05,
    'pitch': 0.12,
    'volume': 0.08,
    'pause': 0.05,
    'tempo': 0.1
}


def transcribe_audio(audio_path):
    model = whisper.load_model("small")
    result = model.transcribe(
        audio_path,
        language="ru",
        word_timestamps=True,
        task="transcribe"
    )
    transcript = result["text"]
    sentences = sent_tokenize(transcript)
    word_timestamps = result["segments"]
    print(f"full text: {transcript}")
    return sentences, word_timestamps


def get_sentences_with_time(sentences,word_timestamps):
  all_words = []
  for x in word_timestamps:
    all_words += x["words"]
  sentences_with_timings = []
  i = -1
  for sen in sentences:
    lsen = sen.split()
    i+=1
    start_ind = i
    i+=len(lsen)-1
    end_ind = i
    if end_ind>len(all_words):
      end_ind = len(all_words)-1
    temp_sen = {"text":sen, "time_start":all_words[start_ind]["start"], "time_end":all_words[end_ind]["end"]}
    sentences_with_timings.append(temp_sen)
  return sentences_with_timings


def extract_audio_features(audio_path, sentences_with_time):
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    # Анализ через Parselmouth (Praat)
    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch()
    formants = sound.to_formant_burg(max_number_of_formants=4)

    features = []
    last_end_time = 0
    for i, sentence in enumerate(sentences_with_time):
        text, start_time, end_time = sentence["text"], sentence["time_start"], sentence["time_end"]
        segment = y[int(start_time*sr):int(end_time*sr)]

        # Анализ тона
        pitch_values = pitch.selected_array['frequency']
        pitch_segment = pitch_values[(pitch.xs() >= start_time) & (pitch.xs() <= end_time)]
        pitch_mean = np.mean(pitch_segment) if len(pitch_segment) > 0 else 0
        pitch_std = np.std(pitch_segment)

        # Форманты
        f1 = formants.get_value_at_time(1, (start_time + end_time)/2)
        f2 = formants.get_value_at_time(2, (start_time + end_time)/2)

        # Темп речи
        words = text.split()
        wpm = len(words)/(end_time - start_time)*60 if (end_time - start_time) > 0 else 0
        n = len(sentences_with_time)
        j = n-i+1+1 if i>n/2 else i+1
        pos = 1-(1-0)*((j-1)/(n/2-1))**2
        features.append({
            "sentence": text,
            #"pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "volume": np.mean(librosa.feature.rms(y=segment)),
            "pause_before": start_time - last_end_time if start_time - last_end_time > 0 else 0,
            "position": pos,
            "tempo_wpm": wpm,
           # "f1_formant": f1,
            #"f2_formant": f2,
           # "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
        })
        last_end_time = end_time
    sr_pause = np.mean([x["pause_before"] for x in features])
    features[0]["pause_before"] = sr_pause
    print(f"{len(sentences_with_time) = }")
    print(f"{features = }")
    return pd.DataFrame(features)




def text_processing(sentences):
    """Текстовая обработка: TF-IDF и TextRank"""
    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=1)).ravel()  # Используем mean вместо sum

    # TextRank
    similarity_matrix = cosine_similarity(tfidf_matrix)  # Косинусная схожесть вместо dot!
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph, alpha=0.85)  # Явно задаем damping_factor
    textrank_scores = np.array([scores[i] for i in range(len(sentences))])

    return tfidf_scores, textrank_scores

def normalize_features(df):
    """Стандартизация признаков"""
    df_norm = df.copy()
    scaler = RobustScaler()
    # audio_features = ['pitch_mean', 'pitch_std', 'volume',
    #                  'f1_formant', 'f2_formant']
    # df_norm[audio_features] = scaler.fit_transform(df[audio_features])

    # features = ['pitch_mean', 'pitch_std', 'volume',
    #                  'f1_formant', 'f2_formant', 'tfidf','textrank']

    features = ['pitch_std', 'volume', 'tfidf','textrank']
    df_norm[features] = scaler.fit_transform(df[features])
    df_norm['pause_before'] = np.log1p(df['pause_before'])
    df_norm['tempo_wpm'] = df['tempo_wpm'] / df['tempo_wpm'].max()
    return df_norm


def combined_scoring(df_norm, weights=DEFAULT_WEIGHTS):
    """Комбинированная оценка важности"""
    df_norm['score'] = (
        weights['tfidf'] * df_norm['tfidf'] +
        weights['textrank'] * df_norm['textrank']  +
        weights['position'] * (1 - df_norm['position']) +
        weights['pitch'] * df_norm['pitch_std'] + #(0.7*df_norm['pitch_mean'] + 0.3*df_norm['pitch_std']) +
        weights['volume'] * df_norm['volume'] +
        weights['pause'] * df_norm['pause_before'] +
        weights['tempo'] * (1 - df_norm['tempo_wpm'])
    )

    print(df_norm)
    return df_norm


def generate_summary(df_scored, num_sentences=5):
    """Генерация итоговой суммаризации"""
    top_sentences = df_scored.sort_values('score', ascending=False).head(num_sentences)
    return " ".join(top_sentences.sort_index()['sentence'].tolist())


def audio_summarization_pipeline(audio_path, summary_length=5):
    """Полный пайплайн суммаризации"""
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1")
    # 1. Транскрибация с автоматическим чанкованием
    sentences,word_timestamps = transcribe_audio(audio_path)
    sentences_with_time = get_sentences_with_time(sentences,word_timestamps)
    # 2. Извлечение признаков
    features = extract_audio_features(audio_path,  sentences_with_time)
    tfidf_scores, textrank_scores = text_processing(sentences)
    features['tfidf'] = tfidf_scores
    features['textrank'] = textrank_scores
    # 4. Нормализация и оценка
    df_norm = normalize_features(features)
    df_scored = combined_scoring(df_norm)
    # 5. Генерация результата
    summary = generate_summary(df_scored, summary_length)
    return sentences, summary, df_scored

    
def do_smth(file):
    audio = AudioSegment.from_wav(file)
    name_of_file = "f"+str(randint(1,10**8))
    audio.export(name_of_file, format="mp3")

    full_text, summary, features = audio_summarization_pipeline(name_of_file)

    return "".join(full_text), summary


demo = gr.Interface(
    do_smth,
    gr.Audio(type="filepath"),
    [
        gr.Textbox(value="", label="Исходный текст"),
        gr.Textbox(value="", label="Сокращенный текст")
    ]
)
demo.launch()