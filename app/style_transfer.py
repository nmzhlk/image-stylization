import os

import streamlit as st
import torch
from PIL import Image

from app.cyclegan import CycleGANInference
from app.nst import NSTInference


@st.cache_resource
def _load_cyclegan(style_name):
    if not style_name or not isinstance(style_name, str):
        raise ValueError(f"Invalid style name: {style_name}")

    try:
        return CycleGANInference(style_name=style_name)
    except FileNotFoundError as e:
        raise ValueError(
            f"Style '{style_name}' not found. Please check available styles."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to load CycleGAN model for style '{style_name}': {e}"
        )


@st.cache_resource
def _load_nst(max_size, num_steps, content_weight, style_weight):
    if not (128 <= max_size <= 2048):
        raise ValueError(f"max_size must be between 128 and 2048, got {max_size}")

    if not (1 <= num_steps <= 2000):
        raise ValueError(f"num_steps must be between 1 and 2000, got {num_steps}")

    if content_weight <= 0:
        raise ValueError(f"content_weight must be positive, got {content_weight}")

    if style_weight <= 0:
        raise ValueError(f"style_weight must be positive, got {style_weight}")

    try:
        return NSTInference(
            max_size=max_size,
            num_steps=num_steps,
            content_weight=content_weight,
            style_weight=style_weight,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load NST model: {e}")


class StyleTransfer:
    MAX_FILE_SIZE = 200 * 1024 * 1024
    MAX_IMAGE_DIMENSION = 8912

    def __init__(self, method: str, **kwargs):
        self.device = None
        self.method = method.lower()

        if self.method not in ["cyclegan", "nst"]:
            raise ValueError(
                f"Unknown style transfer method: {method}. Use 'cyclegan' or 'nst'"
            )

        try:
            if self.method == "cyclegan":
                if "style_name" not in kwargs:
                    raise ValueError("CycleGAN requires 'style_name' parameter")

                self.model = _load_cyclegan(style_name=kwargs["style_name"])
                self.device = self.model.device

            elif self.method == "nst":
                required_params = [
                    "max_size",
                    "num_steps",
                    "content_weight",
                    "style_weight",
                ]
                missing_params = [p for p in required_params if p not in kwargs]
                if missing_params:
                    raise ValueError(f"NST missing parameters: {missing_params}")

                self.model = _load_nst(**{k: kwargs[k] for k in required_params})
                self.device = self.model.device

        except Exception as e:
            raise RuntimeError(f"Failed to initialize {method.upper()} model: {e}")

    def _validate_image(self, image, image_name="image"):
        if image is None:
            raise ValueError(f"{image_name} is None")

        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"File not found: {image}")

            file_size = os.path.getsize(image)
            if file_size > self.MAX_FILE_SIZE:
                raise ValueError(
                    f"File {image} is too large ({file_size / 1024 / 1024:.1f}MB). "
                    f"Maximum size is {self.MAX_FILE_SIZE / 1024 / 1024}MB"
                )

            try:
                img = Image.open(image)
                img.verify()
                img = Image.open(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"Invalid image file {image}: {e}")

            image = img

        elif isinstance(image, Image.Image):
            if image.mode != "RGB":
                try:
                    image = image.convert("RGB")
                except Exception as e:
                    raise ValueError(
                        f"Cannot convert image from {image.mode} to RGB: {e}"
                    )
        else:
            raise TypeError(f"Expected PIL Image or file path, got {type(image)}")

        width, height = image.size
        if width > self.MAX_IMAGE_DIMENSION or height > self.MAX_IMAGE_DIMENSION:
            raise ValueError(
                f"Image dimensions ({width}x{height}) exceed maximum "
                f"({self.MAX_IMAGE_DIMENSION}x{self.MAX_IMAGE_DIMENSION})"
            )

        if width < 16 or height < 16:
            raise ValueError(
                f"Image dimensions ({width}x{height}) are too small (minimum 16x16)"
            )

        if image.getbbox() is None:
            raise ValueError("Image appears to be empty")

        return image

    def _validate_output_path(self, output_path):
        if output_path is None:
            return

        try:
            output_dir = os.path.dirname(output_path) or "."
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            test_file = os.path.join(output_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)

        except Exception as e:
            raise ValueError(f"Cannot write to output path {output_path}: {e}")

    def cleanup(self):
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(
        self,
        content_image,
        style_image=None,
        output_path=None,
        progress_callback=None,
        **kwargs,
    ):
        try:
            content_image = self._validate_image(content_image, "content_image")

            if self.method == "cyclegan":
                self._validate_output_path(output_path)
                try:
                    result, total_time = self.model.transfer_style(
                        image=content_image,
                        output_path=output_path,
                    )
                    return {
                        "image": result,
                        "time": total_time,
                        "method": "cyclegan",
                        "device": self.device,
                        "success": True,
                    }

                except Exception as e:
                    self.cleanup()
                    return {
                        "image": None,
                        "time": 0,
                        "method": "cyclegan",
                        "device": self.device,
                        "success": False,
                        "error": str(e),
                    }

            if self.method == "nst":
                if style_image is None:
                    raise ValueError("NST requires style image")

                style_image = self._validate_image(style_image, "style_image")
                self._validate_output_path(output_path)

                try:
                    result, total_time = self.model.transfer_style(
                        content_image=content_image,
                        style_image=style_image,
                        output_path=output_path,
                        progress_callback=progress_callback,
                    )
                    return {
                        "image": result,
                        "time": total_time,
                        "method": "nst",
                        "device": self.device,
                        "success": True,
                    }

                except Exception as e:
                    self.cleanup()
                    return {
                        "image": None,
                        "time": 0,
                        "method": self.method,
                        "device": self.device,
                        "success": False,
                        "error": str(e),
                    }

        except Exception as e:
            self.cleanup()
            return {
                "image": None,
                "time": 0,
                "method": self.method,
                "device": self.device,
                "success": False,
                "error": str(e),
            }

    def __del__(self):
        self.cleanup()
