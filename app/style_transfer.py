from cyclegan import CycleGANInference
from nst import NSTInference


class StyleTransfer:
    def __init__(self, method: str, **kwargs):
        self.method = method.lower()

        if self.method == "cyclegan":
            self.model = CycleGANInference(**kwargs)

        elif self.method == "nst":
            self.model = NSTInference(**kwargs)

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
                **kwargs,
            )
            return result, None

        if self.method == "nst":
            if style_image is None:
                raise ValueError("NST requires style_image")

            result, total_time = self.model.transfer_style(
                content_image=content_image,
                style_image=style_image,
                output_path=output_path,
                progress_callback=progress_callback,
            )
            return result, total_time
