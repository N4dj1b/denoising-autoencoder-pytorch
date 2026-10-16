"""Dataset loading helpers. The default sklearn Digits dataset is offline-ready."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits


@dataclass
class ImageSplits:
    train: torch.Tensor
    test: torch.Tensor
    description: str


def _limit_rows(images: torch.Tensor, limit: Optional[int]) -> torch.Tensor:
    if limit is None or limit <= 0:
        return images
    return images[: min(limit, len(images))]


def load_image_splits(
    dataset: str = "digits",
    data_dir: str | Path = "data",
    train_limit: Optional[int] = 1200,
    test_limit: Optional[int] = 350,
    seed: int = 42,
    image_size: int = 28,
) -> ImageSplits:
    """Load public grayscale data and return normalized [N, 1, H, W] tensors.

    `digits` is bundled with scikit-learn (1,797 8x8 grayscale images), so it
    works without downloading data. It is resized to 28x28 for the model.
    `mnist` downloads the public 28x28 MNIST dataset on first use.
    """
    generator = torch.Generator().manual_seed(seed)
    dataset = dataset.lower()

    if dataset == "digits":
        source = load_digits()
        images = torch.tensor(source.images, dtype=torch.float32).unsqueeze(1) / 16.0
        images = F.interpolate(
            images, size=(image_size, image_size), mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        permutation = torch.randperm(len(images), generator=generator)
        split_index = int(0.8 * len(images))
        train = images[permutation[:split_index]]
        test = images[permutation[split_index:]]
        description = (
            "scikit-learn Digits (1,797 source images at 8x8, normalized and "
            f"bilinearly resized to {image_size}x{image_size})"
        )
    elif dataset == "mnist":
        try:
            from torchvision.datasets import MNIST
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "The MNIST option requires a working torchvision installation."
            ) from exc
        root = Path(data_dir)
        train_ds = MNIST(root=str(root), train=True, download=True)
        test_ds = MNIST(root=str(root), train=False, download=True)
        train = train_ds.data.float().unsqueeze(1) / 255.0
        test = test_ds.data.float().unsqueeze(1) / 255.0
        train = F.interpolate(
            train, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        test = F.interpolate(
            test, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        train = train[torch.randperm(len(train), generator=generator)]
        test = test[torch.randperm(len(test), generator=generator)]
        description = f"MNIST (28x28 grayscale images resized to {image_size}x{image_size})"
    else:
        raise ValueError("dataset must be either 'digits' or 'mnist'")

    train = _limit_rows(train.contiguous(), train_limit)
    test = _limit_rows(test.contiguous(), test_limit)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Training and test subsets must both contain at least one image")
    return ImageSplits(train=train, test=test, description=description)
