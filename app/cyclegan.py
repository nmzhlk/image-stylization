import functools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    if norm_type == "none":
        return lambda x: nn.Identity()
    raise NotImplementedError(f"Normalization layer {norm_type} not implemented")


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        padding_type="reflect",
        norm_layer=None,
        use_dropout=False,
        use_bias=True,
    ):
        super().__init__()

        conv_block = []
        p = 0

        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"Padding {padding_type} not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, 3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]

        if use_dropout:
            conv_block.append(nn.Dropout(0.5))

        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1

        conv_block += [
            nn.Conv2d(dim, dim, 3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        ngf=64,
        norm_layer=None,
        use_dropout=False,
        n_blocks=9,
        padding_type="reflect",
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = get_norm_layer("instance")

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        for i in range(2):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=use_bias
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 4
        for _ in range(n_blocks):
            model.append(
                ResnetBlock(mult * ngf, padding_type, norm_layer, use_dropout, use_bias)
            )

        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    ngf * mult // 2,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class CycleGANInference:
    _MODEL_CACHE = {}

    def __init__(self, style_name, device=None):
        self.device = self._get_device(device)
        self.style_name = style_name

        base_dir = Path(__file__).parent / "models" / "checkpoints"
        self.model_path = base_dir / style_name / "latest_net_G.pth"

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                f"Available styles: {self._get_available_styles()}"
            )

        self.model = self._load_model_cached()

    def _get_device(self, device):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

        return device

    def _get_available_styles(self):
        base_dir = Path(__file__).parent / "models" / "checkpoints"
        if not base_dir.exists():
            return []

        return [
            d.name
            for d in base_dir.iterdir()
            if d.is_dir() and (d / "latest_net_G.pth").exists()
        ]

    def _load_model_cached(self):
        cache_key = f"{self.style_name}_{self.device}"

        if cache_key not in self._MODEL_CACHE:
            model = self._load_model()
            self._MODEL_CACHE[cache_key] = model

        return self._MODEL_CACHE[cache_key]

    def _load_model(self):
        try:
            model = ResnetGenerator(norm_layer=get_norm_layer("instance"))
            state_dict = torch.load(self.model_path, map_location="cpu")

            if next(iter(state_dict)).startswith("module."):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not any(
                    x in k
                    for x in ["running_mean", "running_var", "num_batches_tracked"]
                )
            }

            model.load_state_dict(state_dict, strict=True)
            model.eval()

            if self.device == "cuda":
                model.cuda()

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_path}: {e}")

    def _optimal_size(self, size, max_size):
        w, h = size
        if max(w, h) <= max_size:
            return size
        ratio = w / h
        if w > h:
            return max_size, int(max_size / ratio)
        return int(max_size * ratio), max_size

    def _enhance(self, image):
        try:
            image = ImageEnhance.Contrast(image).enhance(1.1)
            image = ImageEnhance.Sharpness(image).enhance(1.05)
            image = ImageEnhance.Color(image).enhance(1.08)
            return image
        except Exception:
            return image

    def cleanup(self):
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transfer_style(self, image, output_path=None, max_size=1024, enhance=True):
        start_time = time.time()

        try:
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise TypeError(f"Expected PIL Image, str or Path, got {type(image)}")

            original_size = image.size
            target_size = self._optimal_size(original_size, max_size)

            transform = transforms.Compose(
                [
                    transforms.Resize(
                        target_size, transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            tensor = transform(image).unsqueeze(0)
            if self.device == "cuda":
                tensor = tensor.cuda()

            with torch.no_grad():
                output = self.model(tensor)

            output = output.squeeze().cpu().numpy()
            if output.shape[0] == 1:
                output = np.tile(output, (3, 1, 1))

            output = np.transpose(output, (1, 2, 0))
            output = np.clip((output + 1) * 127.5, 0, 255).astype(np.uint8)
            result = Image.fromarray(output)

            if enhance:
                result = self._enhance(result)

            if target_size != original_size:
                result = result.resize(original_size, Image.Resampling.LANCZOS)

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.suffix.lower() in [".jpg", ".jpeg"]:
                    result.save(output_path, quality=95, subsampling=0, optimize=True)
                else:
                    result.save(output_path, quality=95)

            end_time = time.time()
            total_time = end_time - start_time

            self.cleanup()
            return result, total_time

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Style transfer failed: {e}")

    def __del__(self):
        self.cleanup()
