from PIL import Image

from app_ui.constants import CYCLEGAN_MODELS_ROOT


def get_available_styles():
    if not CYCLEGAN_MODELS_ROOT.exists():
        return []

    return [
        d.name
        for d in CYCLEGAN_MODELS_ROOT.iterdir()
        if (d / "latest_net_G.pth").exists()
    ]


def load_cyclegan_style_preview(style_key):
    if style_key is None:
        return None

    preview_path = CYCLEGAN_MODELS_ROOT / style_key / "preview.jpg"

    if preview_path.exists():
        return Image.open(preview_path).convert("RGB")

    return None


def get_style_display_map():
    from app_ui.constants import STYLE_LABELS

    available_styles = get_available_styles()
    return {
        STYLE_LABELS.get(style, style.replace("_", " ").title()): style
        for style in available_styles
    }
