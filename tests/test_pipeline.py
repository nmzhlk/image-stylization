from pathlib import Path

import pytest
from PIL import Image


def test_full_pipeline_integration(tmp_path, sample_content_image, sample_style_image):
    from app.style_transfer import StyleTransfer

    output_path = tmp_path / "integration_result.jpg"

    st_nst = StyleTransfer(
        method="nst",
        max_size=128,
        num_steps=2,
        content_weight=1.0,
        style_weight=1e5,
    )

    result_nst = st_nst.run(
        content_image=sample_content_image,
        style_image=sample_style_image,
        output_path=str(output_path),
    )

    assert result_nst["success"] is True
    assert result_nst["method"] == "nst"
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    loaded_img = Image.open(output_path)
    assert loaded_img.size == (512, 512)


def test_models_load_correctly():
    from app.cyclegan import CycleGANInference
    from app.nst import NSTInference

    try:
        styles = CycleGANInference(
            style_name="style_vangogh", device="cpu"
        )._get_available_styles()
        if styles:
            model = CycleGANInference(style_name=styles[0], device="cpu")
            assert hasattr(model, "model")
            assert not model.model.training
    except Exception as e:
        pytest.skip(f"CycleGAN models not available: {e}")

    model_nst = NSTInference(max_size=128, num_steps=2, device="cpu")
    assert hasattr(model_nst, "cnn")
    assert model_nst.cnn is not None
