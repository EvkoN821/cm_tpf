import whisper
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

def transcribe_and_process(audio_path):
    """Функция F2: Транскрибация и обработка текста"""
    # Транскрибация
    model = whisper.load_model("small")
    result = model.transcribe(
        audio_path,
        language="ru",
        word_timestamps=True,
        task="transcribe"
    )
    
    # Обработка текста
    transcript = result["text"]
    sentences = sent_tokenize(transcript)
    word_timestamps = result["segments"]
    
    # Извлечение признаков текста
    tfidf_scores, textrank_scores = text_processing(sentences)
    
    return sentences, word_timestamps, tfidf_scores, textrank_scores

def text_processing(sentences):
    """Текстовая обработка: TF-IDF и TextRank"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=1)).ravel()
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph, alpha=0.85)
    textrank_scores = np.array([scores[i] for i in range(len(sentences))])
    
    return tfidf_scores, textrank_scores