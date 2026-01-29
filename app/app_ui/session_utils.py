import streamlit as st


def init_session_state():
    if "result_image" not in st.session_state:
        st.session_state.result_image = None

    if "content_image" not in st.session_state:
        st.session_state.content_image = None

    if "cyclegan_style_label" not in st.session_state:
        st.session_state.cyclegan_style_label = None

    if "nst_params" not in st.session_state:
        st.session_state.nst_params = None

    if "prev_method" not in st.session_state:
        st.session_state.prev_method = None

    if "original_filename" not in st.session_state:
        st.session_state.original_filename = None


def reset_result():
    st.session_state.result_image = None
    st.session_state.content_image = None
