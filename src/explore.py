"""Exploratory data analysis: class distribution and sample image grid."""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import CLASS_NAMES, FIG_DIR, IMG_SIZE, SPLIT_DIR, BATCH_SIZE, SEED


def dataset(split):
    return tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / split,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=(split == "train"),
        seed=SEED if split == "train" else None,
    )


def main():
    # ---- class distribution --------------------------------------------
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = Counter(
            p.parent.name for p in (SPLIT_DIR / split).glob("*/*")
        )

    with open(FIG_DIR / "class_counts.json", "w") as f:
        json.dump({k: dict(v) for k, v in counts.items()}, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, split in zip(axes, ("train", "val", "test")):
        values = [counts[split].get(c, 0) for c in CLASS_NAMES]
        ax.bar(CLASS_NAMES, values, color="steelblue")
        ax.set_title(f"{split} ({sum(values)} images)")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Class distribution per split")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # ---- sample images ---------------------------------------------------
    train = dataset("train")
    images, labels = next(iter(train))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, img, lbl in zip(axes.ravel(), images[:8], labels[:8]):
        ax.imshow(img.numpy().astype("uint8"))
        ax.set_title(CLASS_NAMES[int(lbl)])
        ax.axis("off")
    fig.suptitle("Sample training images")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sample_images.png", dpi=150)
    plt.close(fig)

    total = sum(sum(c.values()) for c in counts.values())
    print(f"Total images: {total}")
    print(json.dumps({k: dict(v) for k, v in counts.items()}, indent=2))
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
