"""Build, train and save the CNN waste classifier.

Pipeline:
  1. Load train/val datasets from data/splits (data augmentation on train).
  2. CNN: 3 x [Conv2D + BatchNorm + MaxPool] -> Dense head with Dropout.
  3. Train with EarlyStopping; experiment via CLI args (--batch-size, --epochs).
  4. Plot accuracy/loss curves, save history and the trained model.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

from common import (BATCH_SIZE, CLASS_NAMES, FIG_DIR, HISTORY_PATH, IMG_SIZE,
                    MODEL_PATH, SPLIT_DIR, SEED)


def build_datasets(batch_size):
    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "train", image_size=IMG_SIZE, batch_size=batch_size,
        label_mode="int", shuffle=True, seed=SEED)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "val", image_size=IMG_SIZE, batch_size=batch_size,
        label_mode="int", shuffle=False)
    # cache + prefetch keeps the CPU-fed pipeline efficient
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (data_aug(x, training=True), y),
        num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds


def build_model():
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)          # pixels 0..255 -> 0..1
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = models.Model(inputs, outputs, name="waste_cnn")
    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(history.history[metric], label=f"train {metric}")
        ax.plot(history.history[f"val_{metric}"], label=f"val {metric}")
        ax.set_xlabel("epoch")
        ax.legend()
        ax.set_title(metric.capitalize())
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_history.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train the waste classification CNN")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    print(f"Training with batch_size={args.batch_size}, epochs={args.epochs}")

    train_ds, val_ds = build_datasets(args.batch_size)
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, callbacks=callbacks)

    plot_history(history)
    HISTORY_PATH.write_text(json.dumps(history.history, indent=2))
    model.save(MODEL_PATH)
    print(f"Model saved to   {MODEL_PATH}")
    print(f"History saved to {HISTORY_PATH}")
    print(f"Plots saved to   {FIG_DIR / 'training_history.png'}")


if __name__ == "__main__":
    main()
