# Portfolio refactor provenance

This repository was created from the uploaded final NLP coursework PDF.

What was preserved:
- dataset description and label mapping
- final preprocessing sequence
- TF-IDF + Naive Bayes pipeline
- TF-IDF + Linear SVM pipeline
- simple neural-network architecture
- Bi-Directional LSTM + GloVe architecture
- reported training settings
- reported final model-comparison accuracies

What was changed:
- notebook-style cells were reorganized into Python modules
- duplicated evaluation cells were consolidated
- generated output paths were standardized
- large data and embedding files were excluded from Git

No additional model family was added.
