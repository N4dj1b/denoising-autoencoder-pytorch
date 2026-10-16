"""Image reconstruction metrics for images whose dynamic range is [0, 1]."""

import math

import torch
import torch.nn.functional as F


def reconstruction_metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(estimate, reference).item()
    mae = F.l1_loss(estimate, reference).item()
    psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    return {"mse": mse, "mae": mae, "psnr_db": psnr}
