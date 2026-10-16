"""Noise models used in the experiments.

These are controlled toy perturbations, not a physical SAR simulator.
"""

import torch


def add_noise(
    clean: torch.Tensor,
    mode: str,
    additive_sigma: float = 0.25,
    gamma_looks: float = 4.0,
) -> torch.Tensor:
    """Return a noisy image tensor, clipped to [0, 1].

    additive_gaussian:
        y = clip(x + sigma * epsilon, 0, 1), epsilon ~ N(0, 1)

    multiplicative_gamma:
        y = clip(x * s, 0, 1), s ~ Gamma(L, rate=L), E[s] = 1.
        This is a simplified speckle-inspired perturbation, not SAR-realistic.
    """
    if mode == "additive_gaussian":
        noisy = clean + additive_sigma * torch.randn(
            clean.shape, dtype=clean.dtype, device=clean.device
        )
    elif mode == "multiplicative_gamma":
        if gamma_looks <= 0:
            raise ValueError("gamma_looks must be greater than zero")
        # Gamma(shape=L, rate=L) has mean 1 and variance 1/L.
        concentration = torch.tensor(gamma_looks, dtype=clean.dtype, device=clean.device)
        speckle_factor = torch.distributions.Gamma(concentration, concentration).sample(clean.shape)
        noisy = clean * speckle_factor
    else:
        raise ValueError(f"Unsupported noise mode: {mode!r}")

    return noisy.clamp_(0.0, 1.0)
