import streamlit as st
from PIL import Image

from app_ui.constants import NST_PRESETS, STYLE_LABELS
from app_ui.session_utils import reset_result
from app_utils.image_utils import load_uploaded_image
from app_utils.style_utils import get_available_styles, load_cyclegan_style_preview


def render_nst_ui():
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
        available_styles = get_available_styles()
        style_display_map = {
            STYLE_LABELS.get(style, style.replace("_", " ").title()): style
            for style in available_styles
        }

        if style_display_map:
            selected_label = st.selectbox(
                "Choose Style",
                list(style_display_map.keys()),
                key="cyclegan_style_select",
                on_change=reset_result,
            )
            selected_style_key = style_display_map[selected_label]

            st.session_state.cyclegan_style_label = selected_label
            st.session_state.selected_style_key = selected_style_key

            preview_image = load_cyclegan_style_preview(selected_style_key)

            if preview_image is not None:
                st.image(
                    preview_image,
                    caption="Style Preview",
                    width="stretch",
                )
            else:
                st.info("Style preview not found")
        else:
            st.warning("No CycleGAN models found")
            selected_label = None
            selected_style_key = None

    return content_file, selected_style_key, selected_label


def render_result_ui(method):
    if st.session_state.result_image is None:
        return

    st.subheader("Result")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            st.session_state.content_image,
            caption="Original",
            width="stretch",
        )

    with col2:
        if method == "nst":
            caption = "NST Stylization"
        else:
            caption = st.session_state.cyclegan_style_label or "Stylized"

        st.image(
            st.session_state.result_image,
            caption=caption,
            width="stretch",
        )

    return True
