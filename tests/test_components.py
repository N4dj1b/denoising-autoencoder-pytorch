import torch

from src.model import DenoisingAutoencoder
from src.noise import add_noise


def test_model_output_shape_and_range():
    model = DenoisingAutoencoder()
    output = model(torch.rand(2, 1, 28, 28))
    assert output.shape == (2, 1, 28, 28)
    assert torch.all((output >= 0) & (output <= 1))


def test_noise_outputs_are_valid():
    clean = torch.rand(4, 1, 28, 28)
    for mode in ("additive_gaussian", "multiplicative_gamma"):
        noisy = add_noise(clean, mode)
        assert noisy.shape == clean.shape
        assert torch.isfinite(noisy).all()
        assert torch.all((noisy >= 0) & (noisy <= 1))


def test_noise_seed_is_reproducible():
    clean = torch.rand(2, 1, 28, 28)
    for mode in ("additive_gaussian", "multiplicative_gamma"):
        torch.manual_seed(123)
        first = add_noise(clean, mode)
        torch.manual_seed(123)
        second = add_noise(clean, mode)
        assert torch.equal(first, second)
