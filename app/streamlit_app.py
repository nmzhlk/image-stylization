import streamlit as st
from app_ui.components import render_cyclegan_ui, render_nst_ui, render_result_ui
from app_ui.session_utils import init_session_state, reset_result
from app_utils.image_utils import (
    create_download_buffer,
    get_download_filename,
    load_uploaded_image,
    make_progress_callback,
)
from style_transfer import StyleTransfer

init_session_state()

st.set_page_config(
    page_title="Image Stylization", page_icon="app/app_ui/icon.png", layout="centered"
)

st.title("Image Stylization")
st.subheader("Neural Style Transfer & CycleGAN")

method_ui = st.selectbox("Select styling method", ["NST", "CycleGAN"])

if st.session_state.get("prev_method") != method_ui:
    reset_result()
st.session_state.prev_method = method_ui

method = method_ui.lower()

if method == "nst":
    content_file, style_file, params = render_nst_ui()
else:
    content_file, selected_style_key, selected_label = render_cyclegan_ui()

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
        st.session_state.original_filename = (
            content_file.name if hasattr(content_file, "name") else None
        )

        if method == "nst":
            if style_file is None:
                st.error("Style image is required for NST")
                st.stop()

            style_img = load_uploaded_image(style_file)
            engine = StyleTransfer(method="nst", **params)

            with st.spinner("Processing..."):
                progress_bar = st.progress(0)
                progress_text = st.empty()

                result_dict = engine.run(
                    content_image=content_path,
                    style_image=style_img,
                    progress_callback=make_progress_callback(
                        params["num_steps"], progress_bar, progress_text
                    ),
                )

            if not result_dict["success"]:
                st.error(f"Error: {result_dict.get('error', 'Unknown error')}")
                st.stop()

            result = result_dict["image"]
            total_time = result_dict["time"]

            st.session_state.nst_params = params
            st.session_state.result_image = result
            st.success(f"Stylization completed in {total_time:.1f} seconds!")
            st.session_state.content_image = content_path

        else:
            if selected_style_key is None:
                st.error("No art style selected")
                st.stop()

            engine = StyleTransfer(method="cyclegan", style_name=selected_style_key)

            with st.spinner("Processing..."):
                content_img = load_uploaded_image(content_file)
                st.session_state.original_filename = (
                    content_file.name if hasattr(content_file, "name") else None
                )

                result_dict = engine.run(content_image=content_img)

            if not result_dict["success"]:
                st.error(f"Error: {result_dict.get('error', 'Unknown error')}")
                st.stop()

            result = result_dict["image"]
            total_time = result_dict["time"]

            st.session_state.result_image = result
            st.success(f"Stylization completed in {total_time:.1f} seconds!")
            st.session_state.content_image = load_uploaded_image(content_file)

    except Exception as e:
        st.error("Model inference failed")
        st.exception(e)
        st.stop()

if render_result_ui(method):
    filename = get_download_filename(
        method=method,
        style_label=st.session_state.get("cyclegan_style_label"),
        style_key=st.session_state.get("selected_style_key"),
        original_filename=st.session_state.get("original_filename"),
    )

    download_buf = create_download_buffer(st.session_state.result_image)
    st.download_button(
        "Download Stylized Image",
        data=download_buf,
        file_name=filename,
        mime="image/png",
    )
