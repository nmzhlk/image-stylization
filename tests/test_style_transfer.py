import pytest
from PIL import Image


def test_style_transfer_nst(sample_content_image, sample_style_image):
    from app.style_transfer import StyleTransfer

    st = StyleTransfer(
        method="nst",
        max_size=128,
        num_steps=2,
        content_weight=1.0,
        style_weight=1e5,
    )

    assert st.method == "nst"
    assert hasattr(st, "model")
    assert st.model.max_size == 128

    result = st.run(content_image=sample_content_image, style_image=sample_style_image)

    assert isinstance(result, dict)
    assert "success" in result
    assert result["success"] is True
    assert "image" in result
    assert isinstance(result["image"], Image.Image)
    assert "time" in result
    assert result["method"] == "nst"


def test_style_transfer_cyclegan(sample_image):
    from app.cyclegan import CycleGANInference
    from app.style_transfer import StyleTransfer

    model = CycleGANInference(style_name="style_vangogh", device="cpu")
    styles = model._get_available_styles()
    if not styles or len(styles) == 0:
        pytest.skip("No CycleGAN models available")

    st = StyleTransfer(method="cyclegan", style_name=styles[0])

    assert st.method == "cyclegan"

    result = st.run(content_image=sample_image, max_size=128)

    assert isinstance(result, dict)
    assert result["method"] == "cyclegan"
    assert result["success"] is True
    assert "image" in result
    assert isinstance(result["image"], Image.Image)


def test_style_transfer_valid_methods():
    from app.style_transfer import StyleTransfer

    st_nst = StyleTransfer(
        method="nst",
        max_size=128,
        num_steps=2,
        content_weight=1.0,
        style_weight=1e5,
    )
    assert st_nst.method == "nst"

    st_nst_upper = StyleTransfer(
        method="NST",
        max_size=128,
        num_steps=2,
        content_weight=1.0,
        style_weight=1e5,
    )
    assert st_nst_upper.method == "nst"

    try:
        from app.cyclegan import CycleGANInference

        model = CycleGANInference(style_name="style_vangogh", device="cpu")
        available_styles = model._get_available_styles()
        if available_styles:
            st_cyclegan = StyleTransfer(
                method="CycleGAN", style_name=available_styles[0]
            )
            assert st_cyclegan.method == "cyclegan"
    except Exception:
        pytest.skip("CycleGAN models not available")


def test_style_transfer_invalid_methods():
    from app.style_transfer import StyleTransfer

    with pytest.raises(ValueError):
        StyleTransfer(method="invalid_method")

    with pytest.raises(ValueError):
        StyleTransfer(method="")

    with pytest.raises(ValueError):
        StyleTransfer(method=" nst")

    with pytest.raises(ValueError):
        StyleTransfer(method="nst ")

    with pytest.raises(RuntimeError):
        StyleTransfer(method="nst")

    with pytest.raises(RuntimeError):
        StyleTransfer(method="cyclegan")


def test_style_transfer_output_path(sample_content_image, sample_style_image, tmp_path):
    from app.style_transfer import StyleTransfer

    output_path = tmp_path / "output.jpg"

    st = StyleTransfer(
        method="nst",
        max_size=128,
        num_steps=2,
        content_weight=1.0,
        style_weight=1e5,
    )

    result = st.run(
        content_image=sample_content_image,
        style_image=sample_style_image,
        output_path=str(output_path),
    )

    assert result["success"] is True
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    img = Image.open(output_path)
    assert img.size == (512, 512)


def test_style_transfer_cache_functions():
    from app.style_transfer import _load_nst

    with pytest.raises(ValueError, match="max_size must be between"):
        _load_nst(max_size=50, num_steps=100, content_weight=1.0, style_weight=1e5)

    with pytest.raises(ValueError, match="max_size must be between"):
        _load_nst(max_size=3000, num_steps=100, content_weight=1.0, style_weight=1e5)

    with pytest.raises(ValueError, match="num_steps must be between"):
        _load_nst(max_size=512, num_steps=0, content_weight=1.0, style_weight=1e5)

    with pytest.raises(ValueError, match="content_weight must be positive"):
        _load_nst(max_size=512, num_steps=100, content_weight=0, style_weight=1e5)

    with pytest.raises(ValueError, match="content_weight must be positive"):
        _load_nst(max_size=512, num_steps=100, content_weight=-1.0, style_weight=1e5)

    with pytest.raises(ValueError, match="style_weight must be positive"):
        _load_nst(max_size=512, num_steps=100, content_weight=1.0, style_weight=0)
