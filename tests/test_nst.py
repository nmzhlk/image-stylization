import pytest
import torch
from PIL import Image


@pytest.fixture
def nst_model():
    from app.nst import NSTInference

    return NSTInference(
        max_size=128,
        num_steps=2,
        content_weight=1.0,
        style_weight=1e5,
        device="cpu",
    )


def test_nst_initialization(nst_model):
    assert nst_model.max_size == 128
    assert nst_model.num_steps == 2
    assert nst_model.device == "cpu"
    assert hasattr(nst_model, "cnn")
    assert hasattr(nst_model, "content_layers")
    assert hasattr(nst_model, "style_layers")


def test_nst_gram_matrix(nst_model):
    test_tensor = torch.randn(1, 3, 4, 4)
    gram = nst_model._gram_matrix(test_tensor)

    assert gram.shape == (1, 3, 3)

    for i in range(3):
        for j in range(3):
            assert torch.allclose(gram[0, i, j], gram[0, j, i], rtol=1e-5)


@pytest.mark.parametrize(
    "input_size,expected",
    [
        ((100, 100), (100, 100)),
        ((200, 100), (128, 64)),
        ((100, 200), (64, 128)),
        ((256, 256), (128, 128)),
        ((300, 150), (128, 64)),
        ((150, 300), (64, 128)),
    ],
)
def test_nst_optimal_size(nst_model, input_size, expected):
    result = nst_model._optimal_size(input_size)
    assert result == expected


@pytest.mark.parametrize(
    "device_input,expected_device",
    [
        ("cpu", "cpu"),
        ("cuda", "cuda" if torch.cuda.is_available() else "cpu"),
        (None, "cuda" if torch.cuda.is_available() else "cpu"),
    ],
)
def test_nst_device(device_input, expected_device):
    from app.nst import NSTInference

    model = NSTInference(device=device_input, max_size=128, num_steps=2)
    assert model.device == expected_device


def test_nst_estimate_time(nst_model, sample_content_image, sample_style_image):
    estimated_time = nst_model.estimate_time(sample_content_image, sample_style_image)

    assert isinstance(estimated_time, float)
    assert estimated_time > 0
    assert estimated_time < 10.0


def test_nst_style_transfer(nst_model, sample_content_image, sample_style_image):
    result, time_taken = nst_model.transfer_style(
        sample_content_image, sample_style_image
    )

    assert isinstance(result, Image.Image)
    assert result.size == (512, 512)
    assert isinstance(time_taken, float)
    assert time_taken > 0


def test_nst_del_method(nst_model):
    nst_model.__del__()

    assert nst_model.max_size == 128
    nst_model.cleanup()


def test_nst_error_handling(nst_model):
    with pytest.raises((TypeError, RuntimeError)):
        nst_model.transfer_style("not_an_image_1", "not_an_image_2")
