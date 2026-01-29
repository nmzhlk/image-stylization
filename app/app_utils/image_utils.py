import io
import os
from datetime import datetime

from PIL import Image


def load_uploaded_image(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")


def make_progress_callback(num_steps, progress_bar, progress_text):
    def progress_callback(step, remaining):
        step = min(step, num_steps)
        progress_bar.progress(step / num_steps)
        progress_text.text(f"Step {step}/{num_steps}, remaining {remaining} sec")

    return progress_callback


def create_download_buffer(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf


def get_download_filename(
    method, style_label=None, style_key=None, original_filename=None
):
    if original_filename and hasattr(original_filename, "name"):
        name = original_filename.name
    elif isinstance(original_filename, str):
        name = original_filename
    else:
        name = "image"

    if "." in name:
        name = name.rsplit(".", 1)[0]
    name = os.path.basename(name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if method == "nst":
        suffix = "nst_stylized"
    elif method == "cyclegan" and style_key:
        if style_key.startswith("style_"):
            style_simple = style_key.replace("style_", "")
        else:
            style_simple = style_key
        suffix = f"{style_simple}_stylized"
    elif method == "cyclegan" and style_label:
        style_simple = style_label.split()[0].lower()
        suffix = f"{style_simple}_stylized"
    else:
        suffix = "stylized"

    return f"{name}_{suffix}_{timestamp}.png"
