from pathlib import Path

CYCLEGAN_MODELS_ROOT = Path("app/models/checkpoints")

STYLE_LABELS = {
    "style_vangogh": "Van Gogh Art Style",
    "style_monet": "Claude Monet Art Style",
    "style_cezanne": "Paul Cezanne Art Style",
    "style_ukiyoe": "Ukiyo-e Japanese Art Style",
}

NST_PRESETS = {
    "Fast Preview": {
        "max_size": 384,
        "num_steps": 100,
        "content_weight": 1.0,
        "style_weight": 5e4,
    },
    "Balanced": {
        "max_size": 512,
        "num_steps": 300,
        "content_weight": 1.0,
        "style_weight": 1e5,
    },
    "Artistic": {
        "max_size": 512,
        "num_steps": 500,
        "content_weight": 1.0,
        "style_weight": 3e5,
    },
    "High Quality": {
        "max_size": 768,
        "num_steps": 700,
        "content_weight": 1.0,
        "style_weight": 5e5,
    },
    "Custom": None,
}
