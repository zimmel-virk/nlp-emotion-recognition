"""Evaluation helpers matching the project's reported metrics and visualizations."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import TARGET_NAMES


def evaluate_classifier(y_true, y_pred, model_name: str, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)

    with (output_dir / f"{model_name}_classification_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=TARGET_NAMES, columns=TARGET_NAMES)
    cm_df.to_csv(output_dir / f"{model_name}_confusion_matrix.csv")

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_df, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=160)
    plt.close()

    return {
        "model": model_name,
        "accuracy": float(accuracy),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }


def plot_training_history(history, title: str, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_model_comparison(metrics, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics)
    df.to_csv(output_dir / "model_comparison.csv", index=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["model"], df["accuracy"])
    plt.ylim(0, 1)
    plt.title("Final Comparison: Statistical vs. Deep Learning Models")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=15, ha="right")
    for bar, value in zip(bars, df["accuracy"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.2%}",
            ha="center",
        )
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close()
