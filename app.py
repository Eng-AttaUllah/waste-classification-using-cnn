"""Gradio app for waste classification.

Tabs:
  1. Image: upload a waste photo -> predicted category + confidence chart.
  2. Video: upload a video -> samples ~1 frame/second, classifies each frame,
     draws the prediction on it and returns an annotated output video.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from common import CLASS_NAMES, MODEL_PATH
from predict_utils import load_model, predict_image

print(f"Loading model from {MODEL_PATH} ...")
model = load_model()
print("Model ready.")


# -------------------------------------------------------------- image tab ----
def classify_image(image):
    if image is None:
        return {}, None
    label, conf, probs = predict_image(model, image)
    return {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}, f"{label} ({conf:.1%})"


# -------------------------------------------------------------- video tab ----
def _reencode_h264(src_path: str) -> str:
    """Re-encode to H.264 (yuv420p) so browsers can actually play the video.

    OpenCV's 'mp4v' codec is MPEG-4 Part 2, which Chrome/Firefox/Safari
    cannot decode in the HTML5 <video> player.
    """
    out_path = src_path.replace(".mp4", "_h264.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         out_path],
        check=True, capture_output=True,
    )
    return out_path


def classify_video(video_path, fps_sample=1.0):
    """Sample frames at fps_sample rate, classify and annotate each one."""
    if video_path is None:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not open the video file.")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / fps_sample)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # unique temp file per run -> avoids Gradio serving a stale cached video
    fd, raw_path = tempfile.mkstemp(suffix="_raw.mp4")
    os.close(fd)
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             src_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise gr.Error("Could not create the output video writer.")

    frame_idx, kept = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label, conf, _ = predict_image(model, rgb)
            text = f"{label} {conf:.0%}"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3, cv2.LINE_AA)
            kept += 1
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Processed {frame_idx} frames, classified {kept} ({fps_sample} fps sampling)")

    if kept == 0:
        raise gr.Error("No frames could be read from the video.")

    h264_path = _reencode_h264(raw_path)
    os.remove(raw_path)          # drop the unplayable intermediate file
    return h264_path


# -------------------------------------------------------------------- UI ----
image_tab = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy", label="Waste image"),
    outputs=[
        gr.Label(num_top_classes=6, label="Prediction"),
        gr.Textbox(label="Top prediction"),
    ],
    title="Waste Classification CNN",
    examples_per_page=6,
)

video_tab = gr.Interface(
    fn=classify_video,
    inputs=[
        gr.Video(label="Waste video"),
        gr.Slider(0.5, 2.0, value=1.0, step=0.5, label="Frames per second to classify"),
    ],
    outputs=gr.Video(label="Annotated video"),
    title="Video Waste Classification",
)

demo = gr.TabbedInterface([image_tab, video_tab], ["Image", "Video"])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
