"""Model definitions reconstructed from the coursework code appendix."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import requests
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
    LSTM,
    TextVectorization,
)
from tensorflow.keras.models import Sequential

from .config import (
    EMBEDDING_DIM,
    RANDOM_STATE,
    SEQUENCE_LENGTH,
    TFIDF_MAX_FEATURES,
    VOCAB_SIZE,
)


def build_naive_bayes() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)),
        ("clf", MultinomialNB()),
    ])


def build_linear_svm() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)),
        ("clf", LinearSVC(dual=False, random_state=RANDOM_STATE, max_iter=10000)),
    ])


def build_text_vectorizer(x_train_tensor) -> TextVectorization:
    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=SEQUENCE_LENGTH,
    )
    vectorizer.adapt(x_train_tensor)
    return vectorizer


def build_simple_neural_network(vectorizer: TextVectorization) -> Sequential:
    model = Sequential([
        vectorizer,
        Embedding(input_dim=VOCAB_SIZE, output_dim=64),
        GlobalAveragePooling1D(),
        Dense(24, activation="relu"),
        Dropout(0.5),
        Dense(6, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def ensure_glove_file(
    glove_file: str | Path = "glove.6B.100d.txt",
    glove_zip: str | Path = "glove.6B.zip",
) -> Path:
    """Download/extract the GloVe file using the workflow documented in the report."""
    glove_file = Path(glove_file)
    glove_zip = Path(glove_zip)
    url = "http://nlp.stanford.edu/data/glove.6B.zip"

    if glove_file.exists():
        return glove_file

    if not glove_zip.exists():
        print("Downloading GloVe embeddings (large archive)...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with glove_zip.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    with zipfile.ZipFile(glove_zip, "r") as archive:
        archive.extract(glove_file.name, path=glove_file.parent)

    return glove_file


def load_glove_embeddings(glove_file: str | Path) -> dict[str, np.ndarray]:
    embeddings_index: dict[str, np.ndarray] = {}
    with Path(glove_file).open(encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype="float32")
    return embeddings_index


def build_embedding_matrix(vectorizer: TextVectorization, embeddings_index):
    vocabulary = vectorizer.get_vocabulary()
    word_index = dict(zip(vocabulary, range(len(vocabulary))))
    num_tokens = len(vocabulary) + 2

    matrix = np.zeros((num_tokens, EMBEDDING_DIM), dtype=np.float32)
    hits = 0
    misses = 0

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1

    return matrix, hits, misses


def build_glove_bilstm(vectorizer: TextVectorization, embedding_matrix: np.ndarray) -> Sequential:
    num_tokens = embedding_matrix.shape[0]

    model = Sequential([
        vectorizer,
        Embedding(
            input_dim=num_tokens,
            output_dim=EMBEDDING_DIM,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False,
        ),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(6, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
