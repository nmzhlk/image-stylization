import pytest
from PIL import Image


@pytest.fixture
def cyclegan_model():
    from app.cyclegan import CycleGANInference

    return CycleGANInference(style_name="style_vangogh", device="cpu")


def test_cyclegan_initialization(cyclegan_model):
    assert hasattr(cyclegan_model, "model")
    assert cyclegan_model.device == "cpu"
    assert cyclegan_model.style_name == "style_vangogh"


def test_cyclegan_invalid_style():
    from app.cyclegan import CycleGANInference

    with pytest.raises(FileNotFoundError):
        CycleGANInference(style_name="not_a_style", device="cpu")


@pytest.mark.parametrize(
    "input_size,max_size,expected",
    [
        ((500, 500), 512, (500, 500)),
        ((1000, 500), 512, (512, 256)),
        ((200, 300), 512, (200, 300)),
        ((1500, 500), 1024, (1024, 341)),
    ],
)
def test_cyclegan_optimal_size(cyclegan_model, input_size, max_size, expected):
    result = cyclegan_model._optimal_size(input_size, max_size)
    assert result == expected


def test_cyclegan_enhance_option(cyclegan_model, sample_content_image):
    result_enhanced, time1 = cyclegan_model.transfer_style(
        sample_content_image, max_size=128, enhance=True
    )

    result_normal, time2 = cyclegan_model.transfer_style(
        sample_content_image, max_size=128, enhance=False
    )

    assert isinstance(result_enhanced, Image.Image)
    assert isinstance(result_normal, Image.Image)
    assert result_enhanced.size == result_normal.size == (512, 512)


def test_cyclegan_style_transfer(cyclegan_model, sample_content_image):
    result, time_taken = cyclegan_model.transfer_style(
        sample_content_image, max_size=128
    )

    assert isinstance(result, Image.Image)
    assert isinstance(time_taken, float)
    assert time_taken > 0


def test_cyclegan_model_caching():
    from app.cyclegan import CycleGANInference

    model_first = CycleGANInference(style_name="style_vangogh", device="cpu")
    model_cached = CycleGANInference(style_name="style_vangogh", device="cpu")

    assert model_first.model is model_cached.model


def test_cyclegan_del_method():
    from app.cyclegan import CycleGANInference

    try:
        model1 = CycleGANInference(style_name="style_vangogh", device="cpu")
    except FileNotFoundError:
        pytest.skip("CycleGAN model not available")
    model1.__del__()

    model2 = CycleGANInference(style_name="style_vangogh", device="cpu")
    assert model2 is not None
    model2.__del__()


def test_cyclegan_error_handling(sample_image):
    from app.cyclegan import CycleGANInference

    try:
        model = CycleGANInference(style_name="style_vangogh", device="cpu")
    except FileNotFoundError:
        pytest.skip("CycleGAN model not available for testing")

    with pytest.raises(RuntimeError, match="Style transfer failed"):
        model.transfer_style(123)

    with pytest.raises(RuntimeError, match="Style transfer failed"):
        model.transfer_style([1, 2, 3])

    with pytest.raises(RuntimeError, match="Style transfer failed"):
        model.transfer_style("/nonexistent/path/image.jpg")

    result, time_taken = model.transfer_style(sample_image, max_size=64)

    assert isinstance(result, Image.Image)
    assert isinstance(time_taken, float)
