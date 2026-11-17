"""Evaluate the trained model on the held-out test set.

Reports accuracy, F1 scores and the confusion matrix; saves everything
to outputs/metrics and outputs/figures.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from common import (BATCH_SIZE, CLASS_NAMES, FIG_DIR, IMG_SIZE, METRIC_DIR,
                    MODEL_PATH, SPLIT_DIR)


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "test", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", shuffle=False)
    class_names = test_ds.class_names

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, target_names=class_names)

    print(f"\nTest accuracy   : {acc:.4f}")
    print(f"F1 (macro)      : {f1_macro:.4f}")
    print(f"F1 (weighted)   : {f1_weighted:.4f}\n")
    print(report)

    METRIC_DIR.joinpath("test_metrics.json").write_text(json.dumps(
        {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted},
        indent=2))
    METRIC_DIR.joinpath("classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (test acc = {acc:.3f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved metrics to {METRIC_DIR} and confusion matrix to {FIG_DIR}")


if __name__ == "__main__":
    main()
