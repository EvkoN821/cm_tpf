import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

DEFAULT_WEIGHTS = {
    'tfidf': 0.31,
    'textrank': 0.29,
    'position': 0.05,
    'pitch': 0.12,
    'volume': 0.08,
    'pause': 0.05,
    'tempo': 0.1
}

def generate_summary(features_df, num_sentences=7, weights=DEFAULT_WEIGHTS):
    """Функция F4: Формирование резюме"""
    # Нормализация признаков
    df_norm = normalize_features(features_df)
    
    # Комбинированная оценка
    df_scored = combined_scoring(df_norm, weights)
    
    # Генерация итоговой суммаризации
    top_sentences = df_scored.sort_values('score', ascending=False).head(num_sentences)
    summary = " ".join(top_sentences.sort_index()['sentence'].tolist())
    
    return summary, df_scored

def normalize_features(df):
    """Стандартизация признаков"""
    df_norm = df.copy()
    scaler = RobustScaler()
    
    features = ['pitch_std', 'volume', 'tfidf', 'textrank']
    df_norm[features] = scaler.fit_transform(df[features])
    df_norm['pause_before'] = np.log1p(df['pause_before'])
    df_norm['tempo_wpm'] = df['tempo_wpm'] / df['tempo_wpm'].max()
    
    return df_norm

def combined_scoring(df_norm, weights):
    """Комбинированная оценка важности"""
    df_norm['score'] = (
        weights['tfidf'] * df_norm['tfidf'] +
        weights['textrank'] * df_norm['textrank'] +
        weights['position'] * (1 - df_norm['position']) +
        weights['pitch'] * df_norm['pitch_std'] +
        weights['volume'] * df_norm['volume'] +
        weights['pause'] * df_norm['pause_before'] +
        weights['tempo'] * (1 - df_norm['tempo_wpm'])
    )
    return df_norm