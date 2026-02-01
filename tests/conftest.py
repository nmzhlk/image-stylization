import pytest
from PIL import Image


@pytest.fixture
def sample_image():
    img = Image.new("RGB", (512, 512), color="red")
    return img


@pytest.fixture
def sample_content_image():
    img = Image.new("RGB", (512, 512), color="green")
    return img


@pytest.fixture
def sample_style_image():
    img = Image.new("RGB", (512, 512), color="blue")
    return img
