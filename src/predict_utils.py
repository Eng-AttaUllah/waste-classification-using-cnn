"""Prediction helpers shared by the Gradio app and tests."""
import numpy as np
import tensorflow as tf

from common import CLASS_NAMES, IMG_SIZE, MODEL_PATH


def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(model, img):
    """img: numpy array (H, W, 3) uint8 or PIL image. Returns (label, confidence, probs)."""
    img = np.asarray(img)
    if img.ndim == 2:                      # grayscale -> RGB
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]                     # drop alpha if present
    resized = tf.image.resize(img, IMG_SIZE).numpy()
    batch = np.expand_dims(resized, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs
