import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.view(-1, 1, 1))
        self.register_buffer("std", std.view(-1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class NSTInference:
    _VGG_CACHE = {}

    def __init__(
        self,
        device=None,
        max_size=512,
        num_steps=300,
        content_weight=1.0,
        style_weight=1e5,
    ):
        self.device = self._get_device(device)
        self.max_size = max_size
        self.num_steps = num_steps
        self.content_weight = content_weight
        self.style_weight = style_weight

        self.cnn = self._load_vgg_model()

        self.content_layers = ["conv_4"]
        self.style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

        self.normalization_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device
        )
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225], device=self.device)

        self._model_cache = None

    def _get_device(self, device):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

        return device

    def _load_vgg_model(self):
        cache_key = ("vgg19", str(self.device))

        if cache_key not in self._VGG_CACHE:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            features = vgg.features

            features = features.to(self.device).eval()
            for param in features.parameters():
                param.requires_grad = False

            self._VGG_CACHE[cache_key] = features

        return self._VGG_CACHE[cache_key]

    @staticmethod
    def _gram_matrix(x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w * b)

    def _optimal_size(self, size):
        w, h = size
        if max(w, h) <= self.max_size:
            return size
        ratio = w / h
        if w > h:
            return self.max_size, int(self.max_size / ratio)
        return int(self.max_size * ratio), self.max_size

    def _load_image(self, img):
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL Image, str or Path, got {type(img)}")

        original_size = img.size
        target_size = self._optimal_size(original_size)

        transform = transforms.Compose(
            [
                transforms.Resize(target_size, Image.Resampling.LANCZOS),
                transforms.ToTensor(),
            ]
        )

        tensor = transform(img).unsqueeze(0).to(self.device)
        return tensor, original_size

    def _build_model(self):
        if self._model_cache is not None:
            return self._model_cache

        normalization = Normalization(self.normalization_mean, self.normalization_std)

        layers = []
        content_layers = {}
        style_layers = {}

        i = 0
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f"conv_{i}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{i}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{i}"
            else:
                continue

            layers.append(layer)

            if name in self.content_layers:
                content_layers[name] = len(layers) - 1
            if name in self.style_layers:
                style_layers[name] = len(layers) - 1

        model = nn.Sequential(normalization, *layers)

        self._model_cache = (model, content_layers, style_layers)
        return self._model_cache

    def _make_closure(
        self,
        target,
        model,
        content_features,
        style_features,
        content_ids,
        style_ids,
    ):
        optimizer = optim.LBFGS([target], max_iter=1)

        def closure():
            optimizer.zero_grad()
            content_loss = 0.0
            style_loss = 0.0

            for name, idx in content_ids.items():
                target_feat = model[: idx + 1](target)
                content_loss += nn.functional.mse_loss(
                    target_feat, content_features[name]
                )

            for name, idx in style_ids.items():
                target_feat = model[: idx + 1](target)
                gram_t = self._gram_matrix(target_feat)
                style_loss += nn.functional.mse_loss(gram_t, style_features[name])

            style_loss = style_loss / len(style_ids)

            total_loss = (
                self.content_weight * content_loss + self.style_weight * style_loss
            )
            total_loss.backward()

            return total_loss

        return closure, optimizer

    def estimate_time(self, content_image, style_image):
        try:
            content, _ = self._load_image(content_image)
            style, _ = self._load_image(style_image)

            model, content_ids, style_ids = self._build_model()

            target = content.clone().requires_grad_(True)

            content_features = {
                name: model[: idx + 1](content).detach()
                for name, idx in content_ids.items()
            }
            style_features = {
                name: self._gram_matrix(model[: idx + 1](style).detach())
                for name, idx in style_ids.items()
            }

            test_closure, test_optimizer = self._make_closure(
                target, model, content_features, style_features, content_ids, style_ids
            )

            start_time = time.time()
            test_optimizer.step(test_closure)
            time_per_step = time.time() - start_time

            estimated_total = time_per_step * self.num_steps
            return estimated_total

        except Exception as e:
            print(f"Error estimating time: {e}")
            return self.num_steps * 2.0

    def cleanup(self):
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transfer_style(
        self, content_image, style_image, output_path=None, progress_callback=None
    ):
        try:
            content, original_size = self._load_image(content_image)
            style, _ = self._load_image(style_image)

            model, content_ids, style_ids = self._build_model()

            content_features = {
                name: model[: idx + 1](content).detach()
                for name, idx in content_ids.items()
            }
            style_features = {
                name: self._gram_matrix(model[: idx + 1](style).detach())
                for name, idx in style_ids.items()
            }

            target = content.clone().requires_grad_(True)

            closure, optimizer = self._make_closure(
                target, model, content_features, style_features, content_ids, style_ids
            )

            start_time = time.time()
            for step in range(1, self.num_steps + 1):
                optimizer.step(closure)

                if progress_callback:
                    elapsed = time.time() - start_time
                    avg_per_step = elapsed / step
                    remaining = max(round(avg_per_step * (self.num_steps - step)), 0)
                    progress_callback(step, remaining)

            with torch.no_grad():
                result = target.detach().squeeze(0).clamp(0, 1).cpu()

            result_pil = transforms.ToPILImage()(result)
            if result_pil.size != original_size:
                result_pil = result_pil.resize(original_size, Image.Resampling.LANCZOS)

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result_pil.save(output_path)

            self.cleanup()

            total_time = time.time() - start_time
            return result_pil, total_time

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Style transfer failed: {e}")

    def __del__(self):
        self.cleanup()
