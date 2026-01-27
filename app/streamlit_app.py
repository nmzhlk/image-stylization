import io
import os
import tempfile
from pathlib import Path

from PIL import Image

import streamlit as st

from style_transfer import StyleTransfer

if "result_image" not in st.session_state:
    st.session_state.result_image = None

if "content_image" not in st.session_state:
    st.session_state.content_image = None

if "cyclegan_style_label" not in st.session_state:
    st.session_state.cyclegan_style_label = None


def reset_result():
    st.session_state.result_image = None
    st.session_state.content_image = None


def load_uploaded_image(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")


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

    col1, col2 = st.columns(2)

    with col1:
        content_file = st.file_uploader(
            "Content Image",
            type=["jpg", "jpeg", "png"],
            key="nst_content",
            on_change=reset_result,
        )
        if content_file:
            st.image(
                Image.open(content_file).convert("RGB"),
                caption="Content Image",
                width="stretch",
            )

    with col2:
        style_file = st.file_uploader(
            "Style Image",
            type=["jpg", "jpeg", "png"],
            key="nst_style",
            on_change=reset_result,
        )
        if style_file:
            st.image(
                Image.open(style_file).convert("RGB"),
                caption="Style Image",
                width="stretch",
            )

    st.subheader("NST Parameters")

    preset = st.selectbox(
        "NST Preset", list(NST_PRESETS.keys()), on_change=reset_result
    )

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
    col_left, col_right = st.columns(2)

    with col_left:
        content_file = st.file_uploader(
            "Content Image",
            ["jpg", "jpeg", "png"],
            key="cyclegan_content",
            on_change=reset_result,
        )

        if content_file:
            original_img = load_uploaded_image(content_file)
            st.image(original_img, caption="Original Image", width="stretch")
        else:
            original_img = None

    with col_right:
        if style_display_map:
            selected_label = st.selectbox(
                "Choose style",
                list(style_display_map.keys()),
                key="cyclegan_style_select",
                on_change=reset_result,
            )
            selected_style_key = style_display_map[selected_label]
            st.session_state.cyclegan_style_label = selected_label
        else:
            st.warning("No CycleGAN models found")
            selected_label = None
            selected_style_key = None

        result_placeholder = st.empty()

        if st.session_state.result_image:
            result_placeholder.image(
                st.session_state.result_image,
                caption=st.session_state.cyclegan_style_label or "Stylized",
                width="stretch",
            )

    return (
        content_file,
        selected_style_key,
        selected_label,
        result_placeholder,
        original_img,
    )


st.set_page_config(page_title="Image Stylization", layout="centered")

st.title("Image Stylization")
st.write("Neural Style Transfer (NST) and CycleGAN")

method_ui = st.selectbox("Select styling method", ["NST", "CycleGAN"])

if st.session_state.get("prev_method") != method_ui:
    reset_result()

st.session_state.prev_method = method_ui

method = method_ui.lower()

if method == "nst":
    content_file, style_file, params = render_nst_ui()
else:
    (
        content_file,
        selected_style_key,
        selected_label,
        result_placeholder,
        original_img,
    ) = render_cyclegan_ui()

if method == "nst" and content_file and style_file:
    tmp_engine = StyleTransfer(method="nst", **params)
    est = tmp_engine.model.estimate_time(
        load_uploaded_image(content_file),
        load_uploaded_image(style_file),
    )
    st.info(f"Estimated time: ~{int(est)} sec")

run = st.button(
    "Stylize!",
    disabled=(
        content_file is None
        or (method == "nst" and style_file is None)
        or (method == "cyclegan" and selected_style_key is None)
    ),
)


if run:
    try:
        if content_file is None:
            st.error("Content image is required")
            st.stop()

        content_path = load_uploaded_image(content_file)

        if method == "nst":
            if style_file is None:
                st.error("Style image is required for NST")
                st.stop()

            style_img = load_uploaded_image(style_file)

            engine = StyleTransfer(method="nst", **params)

            with st.spinner("Processing..."):
                progress_bar = st.progress(0)
                progress_text = st.empty()

                result, total_time = engine.run(
                    content_image=content_path,
                    style_image=style_img,
                    progress_callback=make_progress_callback(
                        params["num_steps"], progress_bar, progress_text
                    ),
                )

            st.session_state.nst_params = params
            st.session_state.result_image = result
            st.session_state.content_image = content_path

        else:
            if selected_style_key is None:
                st.error("No art style selected")
                st.stop()

            engine = StyleTransfer(method="cyclegan", style_name=selected_style_key)

            with st.spinner("Processing..."):
                content_img = load_uploaded_image(content_file)
                result, total_time = engine.run(content_image=content_img)

            st.session_state.result_image = result
            st.session_state.content_image = load_uploaded_image(content_file)

            result_placeholder.image(
                result, caption=st.session_state.cyclegan_style_label, width="stretch"
            )

    except Exception as e:
        st.error("Model inference failed")
        st.exception(e)
        st.stop()


if method == "nst" and st.session_state.result_image is not None:
    st.subheader("Result")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            st.session_state.content_image,
            caption="Original",
            width="stretch",
        )

    with col2:
        st.image(
            st.session_state.result_image,
            caption="NST Stylization",
            width="stretch",
        )

    buf = io.BytesIO()
    st.session_state.result_image.save(buf, format="PNG")
    buf.seek(0)

    st.download_button(
        "Download result",
        data=buf,
        file_name="stylized.png",
        mime="image/png",
    )

if method == "cyclegan" and st.session_state.result_image is not None:
    st.subheader("Result")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            st.session_state.content_image,
            caption="Original",
            width="stretch",
        )

    with col2:
        st.image(
            st.session_state.result_image,
            caption=st.session_state.cyclegan_style_label or "Stylized",
            width="stretch",
        )

    buf = io.BytesIO()
    st.session_state.result_image.save(buf, format="PNG")
    buf.seek(0)

    st.download_button(
        "Download result",
        data=buf,
        file_name="stylized.png",
        mime="image/png",
    )
