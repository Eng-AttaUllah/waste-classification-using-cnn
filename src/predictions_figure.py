"""Generate a figure of model predictions on sample test images."""
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import CLASS_NAMES, FIG_DIR, IMG_SIZE, MODEL_PATH, SPLIT_DIR


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    rng = random.Random(42)

    n_per_class = 2
    samples = []
    for true_cls in CLASS_NAMES:
        files = sorted((SPLIT_DIR / "test" / true_cls).glob("*"))
        rng.shuffle(files)
        for f in files[:n_per_class]:
            img = tf.keras.utils.load_img(f, target_size=IMG_SIZE)
            samples.append((np.asarray(img), true_cls, f.name))

    fig, axes = plt.subplots(4, 3, figsize=(11, 13))
    correct = 0
    for ax, (img, true_cls, fname) in zip(axes.ravel(), samples):
        batch = np.expand_dims(img, axis=0)
        probs = model.predict(batch, verbose=0)[0]
        pred = CLASS_NAMES[int(np.argmax(probs))]
        conf = probs.max()
        correct += int(pred == true_cls)
        ax.imshow(img)
        color = "green" if pred == true_cls else "red"
        ax.set_title(f"true: {true_cls}\npred: {pred} ({conf:.0%})",
                     color=color, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Predictions on {len(samples)} test images "
                 f"({correct}/{len(samples)} correct)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "test_predictions.png", dpi=150)
    print(f"Saved {FIG_DIR / 'test_predictions.png'} — {correct}/{len(samples)} correct")


if __name__ == "__main__":
    main()
