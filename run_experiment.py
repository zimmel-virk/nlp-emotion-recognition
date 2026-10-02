"""Run the reconstructed NLP coursework experiment.

The workflow intentionally preserves the methodology documented in the coursework
appendix, including using the held-out test split as validation_data for the neural
models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from src.evaluation import evaluate_classifier, plot_training_history, save_model_comparison
from src.models import (
    build_embedding_matrix,
    build_glove_bilstm,
    build_linear_svm,
    build_naive_bayes,
    build_simple_neural_network,
    build_text_vectorizer,
    ensure_glove_file,
    load_glove_embeddings,
)
from src.preprocessing import load_dataset, preprocess_and_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/emotions.csv")
    parser.add_argument("--glove", default="glove.6B.100d.txt")
    parser.add_argument("--skip-glove", action="store_true")
    parser.add_argument("--output", default="results/generated")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(args.data)
    print(f"Rows: {len(df):,}")
    print(df["label_name"].value_counts())

    print("\nRunning preprocessing and stratified split...")
    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    metrics = []

    print("\nTraining Multinomial Naive Bayes...")
    nb = build_naive_bayes()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    metrics.append(evaluate_classifier(y_test, nb_pred, "naive_bayes", output_dir))

    print("\nTraining Linear SVM...")
    svm = build_linear_svm()
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    metrics.append(evaluate_classifier(y_test, svm_pred, "linear_svm", output_dir))

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    y_train_cat = to_categorical(y_train_enc)
    y_test_cat = to_categorical(y_test_enc)

    X_train_safe = tf.convert_to_tensor(list(X_train))
    X_test_safe = tf.convert_to_tensor(list(X_test))

    vectorizer = build_text_vectorizer(X_train_safe)

    print("\nTraining simple neural network...")
    simple_nn = build_simple_neural_network(vectorizer)
    history = simple_nn.fit(
        X_train_safe,
        y_train_cat,
        epochs=15,
        batch_size=32,
        validation_data=(X_test_safe, y_test_cat),
        verbose=1,
    )
    simple_probs = simple_nn.predict(X_test_safe, verbose=0)
    simple_pred = np.argmax(simple_probs, axis=1)
    metrics.append(
        evaluate_classifier(
            y_test_enc,
            simple_pred,
            "simple_neural_network",
            output_dir,
        )
    )
    plot_training_history(
        history,
        "Simple Neural Network Training History",
        output_dir / "simple_neural_network_history.png",
    )

    if not args.skip_glove:
        print("\nPreparing GloVe embeddings...")
        glove_path = ensure_glove_file(args.glove)
        embeddings_index = load_glove_embeddings(glove_path)
        embedding_matrix, hits, misses = build_embedding_matrix(
            vectorizer,
            embeddings_index,
        )
        print(f"Vocabulary coverage: {hits} hits, {misses} misses")

        print("\nTraining Bi-Directional LSTM + GloVe...")
        glove_model = build_glove_bilstm(vectorizer, embedding_matrix)
        history_glove = glove_model.fit(
            X_train_safe,
            y_train_cat,
            epochs=10,
            batch_size=32,
            validation_data=(X_test_safe, y_test_cat),
            verbose=1,
        )
        glove_probs = glove_model.predict(X_test_safe, verbose=0)
        glove_pred = np.argmax(glove_probs, axis=1)
        metrics.append(
            evaluate_classifier(
                y_test_enc,
                glove_pred,
                "bilstm_glove",
                output_dir,
            )
        )
        plot_training_history(
            history_glove,
            "Bi-Directional LSTM + GloVe Training History",
            output_dir / "bilstm_glove_history.png",
        )

    save_model_comparison(metrics, output_dir)

    print("\nModel comparison:")
    for row in metrics:
        print(
            f"{row['model']:<24} "
            f"accuracy={row['accuracy']:.4f} "
            f"weighted_f1={row['weighted_f1']:.4f}"
        )
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
