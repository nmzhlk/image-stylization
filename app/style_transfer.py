import streamlit as st

from cyclegan import CycleGANInference
from nst import NSTInference


@st.cache_resource
def _load_cyclegan(style_name: str):
    return CycleGANInference(style_name=style_name)


@st.cache_resource
def _load_nst(
    max_size: int,
    num_steps: int,
    content_weight: float,
    style_weight: float,
):
    return NSTInference(
        max_size=max_size,
        num_steps=num_steps,
        content_weight=content_weight,
        style_weight=style_weight,
    )


class StyleTransfer:
    def __init__(self, method: str, **kwargs):
        self.method = method.lower()

        if self.method == "cyclegan":
            self.model = _load_cyclegan(style_name=kwargs["style_name"])

        elif self.method == "nst":
            self.model = _load_nst(
                max_size=kwargs["max_size"],
                num_steps=kwargs["num_steps"],
                content_weight=kwargs["content_weight"],
                style_weight=kwargs["style_weight"],
            )

        else:
            raise ValueError(f"Unknown style transfer method: {method}")

    def run(
        self,
        content_image,
        style_image=None,
        output_path=None,
        progress_callback=None,
        **kwargs,
    ):
        if self.method == "cyclegan":
            result = self.model.transfer_style(
                image=content_image,
                output_path=output_path,
            )
            return result, None

        if self.method == "nst":
            if style_image is None:
                raise ValueError("NST requires style image")

            result, total_time = self.model.transfer_style(
                content_image=content_image,
                style_image=style_image,
                output_path=output_path,
                progress_callback=progress_callback,
            )
            return result, total_time
