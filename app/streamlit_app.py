import os
import tempfile
from pathlib import Path

from PIL import Image

import streamlit as st

from style_transfer import StyleTransfer


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    image = Image.open(uploaded_file).convert("RGB")
    image.save(tmp.name)
    return Path(tmp.name)


def make_progress_callback(num_steps, progress_bar, progress_text):
    def progress_callback(step, remaining):
        step = min(step, num_steps)
        progress_bar.progress(step / num_steps)
        progress_text.text(f"Step {step}/{num_steps}, remaining {remaining} sec")

    return progress_callback


CYCLEGAN_MODELS_ROOT = Path("app/models/checkpoints")

STYLE_LABELS = {
    "style_vangogh": "Van Gogh Art Style",
    "style_monet": "Claude Monet Art Style",
    "style_cezanne": "Paul Cezanne Art Style",
    "style_ukiyoe": "Ukiyo-e Japanese Art Style",
}

available_styles = [
    d.name for d in CYCLEGAN_MODELS_ROOT.iterdir() if (d / "latest_net_G.pth").exists()
]

style_display_map = {
    STYLE_LABELS.get(style, style.replace("_", " ").title()): style
    for style in available_styles
}


def render_nst_ui():
    content_file = None
    style_file = None

    NST_PRESETS = {
        "Fast Preview": {
            "max_size": 384,
            "num_steps": 100,
            "content_weight": 1.0,
            "style_weight": 5e4,
        },
        "Balanced": {
            "max_size": 512,
            "num_steps": 300,
            "content_weight": 1.0,
            "style_weight": 1e5,
        },
        "Artistic": {
            "max_size": 512,
            "num_steps": 500,
            "content_weight": 1.0,
            "style_weight": 3e5,
        },
        "High Quality": {
            "max_size": 768,
            "num_steps": 700,
            "content_weight": 1.0,
            "style_weight": 5e5,
        },
        "Custom": None,
    }

    preset = st.selectbox("NST Preset", list(NST_PRESETS.keys()))

    if preset != "Custom":
        preset_values = NST_PRESETS[preset]
        max_size = preset_values["max_size"]
        num_steps = preset_values["num_steps"]
        content_weight = preset_values["content_weight"]
        style_weight = preset_values["style_weight"]
        custom = False
    else:
        max_size = 512
        num_steps = 300
        content_weight = 1.0
        style_weight = 1e5
        custom = True

    col1, col2 = st.columns(2)

    with col1:
        content_file = st.file_uploader(
            "Content Image",
            type=["jpg", "jpeg", "png"],
            key="nst_content",
        )
        if content_file:
            st.image(
                Image.open(content_file).convert("RGB"),
                caption="Content Image",
                width=300,
            )

    with col2:
        style_file = st.file_uploader(
            "Style Image",
            type=["jpg", "jpeg", "png"],
            key="nst_style",
        )
        if style_file:
            st.image(
                Image.open(style_file).convert("RGB"),
                caption="Style Image",
                width=300,
            )

    st.subheader("NST Parameters")

    p1, p2 = st.columns(2)

    with p1:
        max_size = st.slider(
            "Max image size",
            256,
            1024,
            max_size,
            step=64,
            disabled=not custom,
        )
        content_weight = st.number_input(
            "Content weight",
            value=content_weight,
            format="%.2f",
            disabled=not custom,
        )

    with p2:
        num_steps = st.slider(
            "Optimization steps",
            50,
            800,
            num_steps,
            step=50,
            disabled=not custom,
        )
        style_weight = st.number_input(
            "Style weight",
            value=style_weight,
            format="%.1e",
            disabled=not custom,
        )

    return (
        content_file,
        style_file,
        {
            "max_size": max_size,
            "num_steps": num_steps,
            "content_weight": content_weight,
            "style_weight": style_weight,
        },
    )


def render_cyclegan_ui():
    content_file = None
    selected_style_key = None
    selected_label = None

    content_file = st.file_uploader(
        "Content Image",
        type=["jpg", "jpeg", "png"],
        key="cyclegan_content",
    )
    if content_file:
        st.image(
            Image.open(content_file).convert("RGB"),
            caption="Content Image",
            width=300,
        )

        st.subheader("Art Style")

        if not style_display_map:
            st.warning("No CycleGAN models found")
        else:
            selected_label = st.selectbox(
                "Choose style",
                list(style_display_map.keys()),
            )
            selected_style_key = style_display_map[selected_label]

    return content_file, selected_style_key, selected_label


st.set_page_config(page_title="Image Stylization", layout="centered")

st.title("Image Stylization")
st.write("Neural Style Transfer (NST) and CycleGAN")

method_ui = st.selectbox("Select styling method", ["NST", "CycleGAN"])
method = method_ui.lower()

content_file = None
style_file = None
params = {}
selected_style_key = None
selected_label = None

if method == "nst":
    content_file, style_file, params = render_nst_ui()
else:
    content_file, selected_style_key, selected_label = render_cyclegan_ui()

run = st.button("Stylize!")

if run:
    if content_file is None:
        st.error("Content image is required")
        st.stop()

    content_path = save_uploaded_file(content_file)
    style_path = None

    try:
        if method == "nst":
            if style_file is None:
                st.error("Style image is required for NST")
                st.stop()

            style_path = save_uploaded_file(style_file)

            engine = StyleTransfer(
                method="nst",
                **params,
            )

            with st.spinner("Processing..."):
                progress_bar = st.progress(0)
                progress_text = st.empty()

                result, total_time = engine.run(
                    content_image=content_path,
                    style_image=style_path,
                    progress_callback=make_progress_callback(
                        params["num_steps"],
                        progress_bar,
                        progress_text,
                    ),
                )

            st.subheader("Result")
            st.image(result, width=450)
            st.success(f"Done! Time: {total_time:.1f} sec")

        else:
            if selected_style_key is None:
                st.error("No art style selected")
                st.stop()

            engine = StyleTransfer(
                method="cyclegan",
                style_name=selected_style_key,
            )

            with st.spinner("Processing..."):
                result, total_time = engine.run(content_image=content_path)

            st.subheader("Result")
            st.image(result, caption=selected_label, width=450)
            st.success("Done!")

    finally:
        try:
            os.remove(content_path)
            if style_path:
                os.remove(style_path)
        except Exception:
            pass
