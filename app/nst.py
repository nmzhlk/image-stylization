import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.view(-1, 1, 1))
        self.register_buffer("std", std.view(-1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class NSTInference:
    def __init__(
        self,
        device=None,
        max_size=512,
        num_steps=300,
        content_weight=1.0,
        style_weight=1e5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_size = max_size
        self.num_steps = num_steps
        self.content_weight = content_weight
        self.style_weight = style_weight

        self.cnn = (
            models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            .features.to(self.device)
            .eval()
        )

        for p in self.cnn.parameters():
            p.requires_grad = False

        self.content_layers = ["conv_4"]
        self.style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

        self.normalization_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device
        )
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225], device=self.device)

    @staticmethod
    def _gram_matrix(x):
        b, c, h, w = x.size()
        features = x.view(c, h * w)
        gram = features @ features.t()
        return gram / (c * h * w)

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
                name = f"layer_{i}"

            layers.append(layer)

            if name in self.content_layers:
                content_layers[name] = len(layers) - 1
            if name in self.style_layers:
                style_layers[name] = len(layers) - 1

        model = nn.Sequential(normalization, *layers)
        return model, content_layers, style_layers

    def estimate_time(self, content_image, style_image):
        content, _ = self._load_image(content_image)
        style, _ = self._load_image(style_image)

        model, content_ids, style_ids = self._build_model()
        model.to(self.device).eval()

        target = content.clone().requires_grad_(True)
        optimizer = optim.LBFGS([target])

        content_features = {
            name: model[: idx + 1](content).detach()
            for name, idx in content_ids.items()
        }
        style_features = {
            name: self._gram_matrix(model[: idx + 1](style).detach())
            for name, idx in style_ids.items()
        }

        step = [0]

        start = time.time()

        def closure():
            optimizer.zero_grad()
            content_loss = 0.0
            style_loss = 0.0

            for name, idx in content_ids.items():
                target_feat = model[: idx + 1](target)
                content_loss += torch.mean((target_feat - content_features[name]) ** 2)

            for name, idx in style_ids.items():
                target_feat = model[: idx + 1](target)
                gram_t = self._gram_matrix(target_feat)
                style_loss += torch.mean((gram_t - style_features[name]) ** 2)

            loss = self.content_weight * content_loss + self.style_weight * style_loss
            loss.backward()
            step[0] += 1
            return loss

        optimizer.step(closure)

        end = time.time()
        time_per_step = end - start
        estimated_total = time_per_step * self.num_steps
        return estimated_total

    def transfer_style(
        self,
        content_image,
        style_image,
        output_path=None,
        progress_callback=None,
    ):
        content, original_size = self._load_image(content_image)
        style, _ = self._load_image(style_image)

        model, content_ids, style_ids = self._build_model()
        model.to(self.device).eval()

        content_features = {
            name: model[: idx + 1](content).detach()
            for name, idx in content_ids.items()
        }
        style_features = {
            name: self._gram_matrix(model[: idx + 1](style).detach())
            for name, idx in style_ids.items()
        }

        target = content.clone().requires_grad_(True)
        optimizer = optim.LBFGS([target])

        step = [0]
        start_time = time.time()
        last_time = start_time

        while step[0] <= self.num_steps:

            def closure():
                optimizer.zero_grad()

                content_loss = 0.0
                style_loss = 0.0

                for name, idx in content_ids.items():
                    target_feat = model[: idx + 1](target)
                    content_loss += torch.mean(
                        (target_feat - content_features[name]) ** 2
                    )

                for name, idx in style_ids.items():
                    target_feat = model[: idx + 1](target)
                    gram_t = self._gram_matrix(target_feat)
                    style_loss += torch.mean((gram_t - style_features[name]) ** 2)

                loss = (
                    self.content_weight * content_loss + self.style_weight * style_loss
                )
                loss.backward()
                step[0] += 1

                if progress_callback is not None:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    avg_per_step = elapsed / step[0]
                    remaining = avg_per_step * (self.num_steps - step[0])
                    progress_callback(step[0], remaining)

                return loss

            optimizer.step(closure)

        result = target.detach().squeeze(0).clamp(0, 1).cpu()
        result = transforms.ToPILImage()(result)

        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)

        total_time = time.time() - start_time
        return result, total_time


def test_nst_inference():
    content_image = Path("data/input/test.jpg")
    style_image = Path("data/input/style.jpg")

    output_dir = Path("data/output/nst")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "nst_result.jpg"

    model = NSTInference(
        max_size=512,
        num_steps=50,
        content_weight=1.0,
        style_weight=1e5,
    )

    def progress_callback(step, remaining_sec):
        print(f"Step {step}/{model.num_steps}, remaining: {remaining_sec:.1f} sec")

    result, total_time = model.transfer_style(
        content_image=content_image,
        style_image=style_image,
        output_path=output_path,
        progress_callback=progress_callback,
    )

    print(f"Result size: {result.size}, total time: {total_time:.1f} sec")


if __name__ == "__main__":
    test_nst_inference()
