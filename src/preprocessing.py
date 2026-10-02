"""Text preprocessing reconstructed from the coursework code appendix."""

from __future__ import annotations

import html
import re
import string
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from .config import EMOTION_MAP, RANDOM_STATE, TEST_SIZE


CONTRACTIONS = {
    "im": "i am",
    "i'm": "i am",
    "dont": "do not",
    "don't": "do not",
    "cant": "cannot",
    "can't": "cannot",
    "wont": "will not",
    "won't": "will not",
    "ive": "i have",
    "i've": "i have",
    "id": "i would",
    "i'd": "i would",
    "youre": "you are",
    "you're": "you are",
    "thats": "that is",
    "that's": "that is",
    "didnt": "did not",
    "didn't": "did not",
    "wouldnt": "would not",
    "wouldn't": "would not",
}


def download_nltk_resources() -> None:
    """Download the NLTK assets named in the coursework appendix."""
    for resource in ("punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"):
        nltk.download(resource, quiet=True)


def expand_contractions(text: str) -> str:
    """Expand the contraction mapping used in the original coursework."""
    for word, replacement in CONTRACTIONS.items():
        text = text.replace(word, replacement)
    return text


def remove_elongated(text: str) -> str:
    """Reduce any character repeated three or more times to two repeats."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def super_clean_text(text: str) -> str:
    """Apply the final 'super advanced' preprocessing pipeline from the report."""
    text = html.unescape(str(text))
    text = text.lower()
    text = expand_contractions(text)

    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"\d+", "", text)

    text = remove_elongated(text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for word in tokens:
        if word not in stop_words:
            clean_tokens.append(lemmatizer.lemmatize(word))

    return " ".join(clean_tokens)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the expected emotions CSV and attach human-readable label names."""
    path = Path(path)
    df = pd.read_csv(path)

    required = {"text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["label_name"] = df["label"].map(EMOTION_MAP)
    return df


def preprocess_and_split(df: pd.DataFrame):
    """Clean the text and perform the documented stratified 80/20 split."""
    download_nltk_resources()

    work = df.copy()
    work["processed_text"] = work["text"].apply(super_clean_text)

    return train_test_split(
        work["processed_text"],
        work["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=work["label"],
    )
