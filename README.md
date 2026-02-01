# Image Stylization [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nmzhlk-image-stylization.streamlit.app/)

Transfer styles between images using neural networks!

![CI badge](https://github.com/nmzhlk/image-stylization/actions/workflows/ci.yml/badge.svg)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=fff)](#)
[![Pytest](https://img.shields.io/badge/Pytest-fff?logo=pytest&logoColor=000)](#)

Try the app now → https://nmzhlk-image-stylization.streamlit.app/

## Features
- **Neural Style Transfer (NST)** – transfer styles from any image
- **NST Modes**: Fast Preview, Balanced, Artistic, High Quality
- **CycleGAN** – transform images into famous art styles
- **CUDA recommended**, but **CPU is also supported** (with longer processing time)
- **Progress Bar** & estimated processing time
- **Download** stylized images with one click

## Quick Start

### Via a Docker container
```bash
# 1. Clone the repository
git clone https://github.com/nmzhlk/image-stylization.git
cd image-stylization

# 2. Compose and start the app
docker-compose up -d

# 3. Open the Streamlit app in your browser
# http://localhost:8501
```

### Manual start
```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app/streamlit_app.py
```

## How to Use
1. **Select stylization method**: NST or CycleGAN
2. **Upload your content image**
    - **For NST**: Also upload a style image
    - **For CycleGAN**: Choose an art style
3. **Adjust NST parameters**: Image size, steps, weights
4. **Stylize!** & download the stylized image

## Tips for Best Results
- **NST Resolution Balance**: Use presets for processing images. Higher resolutions require significantly more VRAM & Time.
- **NST Weights**: If the result image looks messy, increase the *Content Weight*. If it barely changed, increase the *Style Weight*.
- **CycleGAN limitations**: These models are trained on specific domains. For example, the *Ukiyo-e* model works best on landscapes and portraits with clear outlines.

## Models
- **Neural Style Transfer (NST)**: VGG19 pretrained on ImageNet.
- **CycleGAN**: ResNet Generator with 9 residual blocks.
- **Available CycleGAN styles**: Van Gogh, Monet, Cezanne, Ukiyo-e – using pretrained weights.

### Neural Style Transfer (NST)
The core algorithm is based on the Gatys et al. approach. It uses a **VGG19** backbone to extract:
- **Content features** from deeper layers (`conv4_2`).
- **Style features** from multiple layers (`conv1_1` through `conv5_1`) via **Gram Matrices**, which capture patterns regardless of spatial structure.

### CycleGAN
For artistic transformations (like Monet or Van Gogh), we use a **Cycle-Consistent Adversarial Network**. 
- **Architecture**: A ResNet-based Generator with 9 residual blocks for high-quality feature mapping.
- **How it works**: CycleGAN learns the underlying distribution of an artist's style, allowing for more semantic stylization.
