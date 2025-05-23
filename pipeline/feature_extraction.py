import librosa
import parselmouth
import numpy as np
import pandas as pd

def extract_features(audio_path, sentences, word_timestamps, tfidf_scores, textrank_scores):
    """Функция F3: Извлечение признаков"""
    y, sr, sound = preprocess_audio(audio_path)
    sentences_with_time = get_sentences_with_time(sentences, word_timestamps)
    
    pitch = sound.to_pitch()
    formants = sound.to_formant_burg(max_number_of_formants=4)
    
    features = []
    last_end_time = 0
    for i, sentence in enumerate(sentences_with_time):
        text, start_time, end_time = sentence["text"], sentence["time_start"], sentence["time_end"]
        segment = y[int(start_time*sr):int(end_time*sr)]
        
        # Анализ аудио-признаков
        pitch_values = pitch.selected_array['frequency']
        pitch_segment = pitch_values[(pitch.xs() >= start_time) & (pitch.xs() <= end_time)]
        pitch_std = np.std(pitch_segment) if len(pitch_segment) > 0 else 0
        
        # Сбор всех признаков
        features.append({
            "sentence": text,
            "pitch_std": pitch_std,
            "volume": np.mean(librosa.feature.rms(y=segment)),
            "pause_before": start_time - last_end_time if start_time - last_end_time > 0 else 0,
            "position": calculate_position(i, len(sentences)),
            "tempo_wpm": len(text.split())/(end_time - start_time)*60 if (end_time - start_time) > 0 else 0,
            "tfidf": tfidf_scores[i],
            "textrank": textrank_scores[i]
        })
        last_end_time = end_time
    
    # Обработка пауз
    sr_pause = np.mean([x["pause_before"] for x in features])
    features[0]["pause_before"] = sr_pause
    
    return pd.DataFrame(features)

def calculate_position(i, n):
    """Вычисление позиционной важности"""
    j = n-i+1+1 if i>n/2 else i+1
    return 1-(1-0)*((j-1)/(n/2-1))**2

def get_sentences_with_time(sentences, word_timestamps):
    """Сопоставление предложений с временными метками"""
    all_words = []
    for x in word_timestamps:
        all_words += x["words"]
    
    sentences_with_timings = []
    i = -1
    for sen in sentences:
        lsen = sen.split()
        i += 1
        start_ind = i
        i += len(lsen)-1
        end_ind = i
        if end_ind > len(all_words):
            end_ind = len(all_words)-1
        temp_sen = {
            "text": sen, 
            "time_start": all_words[start_ind]["start"], 
            "time_end": all_words[end_ind]["end"]
        }
        sentences_with_timings.append(temp_sen)
    return sentences_with_timings