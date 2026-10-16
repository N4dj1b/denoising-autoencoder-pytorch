"""Train and evaluate four denoising-autoencoder conditions.

Experiment matrix: two noise models x two reconstruction losses (MSE and L1).
Example: python -m src.train --dataset digits --epochs 6
"""

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_image_splits
from src.metrics import reconstruction_metrics
from src.model import DenoisingAutoencoder
from src.noise import add_noise

NOISE_MODES = ("additive_gaussian", "multiplicative_gamma")
LOSSES = ("mse", "l1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("digits", "mnist"), default="digits")
    parser.add_argument("--data-dir", default="data", help="Where downloaded datasets are stored")
    parser.add_argument("--output-dir", default="results", help="Directory for metrics, figures, and weights")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int, default=1200, help="Use 0 for all available training images")
    parser.add_argument("--test-limit", type=int, default=350, help="Use 0 for all available test images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--additive-sigma", type=float, default=0.25)
    parser.add_argument("--gamma-looks", type=float, default=4.0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loss(name: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if name == "mse":
        return F.mse_loss(prediction, target)
    if name == "l1":
        return F.l1_loss(prediction, target)
    raise ValueError(f"Unsupported loss: {name}")


def fit_model(
    train_images: torch.Tensor,
    noise_mode: str,
    loss_name: str,
    args,
    device: torch.device,
    model_dir: Path,
):
    # Resetting the seed gives each condition the same initialization and data-order seed.
    set_seed(args.seed)
    model = DenoisingAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader_generator = torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(
        TensorDataset(train_images),
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        drop_last=False,
    )
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for (clean_cpu,) in loader:
            clean = clean_cpu.to(device)
            noisy = add_noise(
                clean,
                noise_mode,
                additive_sigma=args.additive_sigma,
                gamma_looks=args.gamma_looks,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(noisy)
            loss = get_loss(loss_name, prediction, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(clean)
            seen += len(clean)

        epoch_loss = total_loss / seen
        history.append({
            "noise": noise_mode,
            "loss": loss_name,
            "epoch": epoch,
            "train_loss": epoch_loss,
        })
        print(f"[{noise_mode:20s} | {loss_name.upper():3s}] epoch {epoch:02d}/{args.epochs}: train_loss={epoch_loss:.6f}")

    model_path = model_dir / f"dae_{noise_mode}_{loss_name}.pt"
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "model_class": "DenoisingAutoencoder",
            "noise_mode": noise_mode,
            "loss": loss_name,
            "seed": args.seed,
            "epochs": args.epochs,
            "image_size": 28,
        },
        model_path,
    )
    model.to(device).eval()
    return model, history, time.time() - start, str(model_path)


def score_condition(model, clean, noisy, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(noisy), 256):
            batch = noisy[start : start + 256].to(device)
            outputs.append(model(batch).cpu())
    denoised = torch.cat(outputs, dim=0)
    baseline = reconstruction_metrics(clean, noisy.cpu())
    denoised_metrics = reconstruction_metrics(clean, denoised)
    return baseline, denoised_metrics, denoised


def save_figures(clean_test, noisy_by_mode, reconstructions, histories, output_dir):
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    count = min(6, len(clean_test))
    column_specs = [
        ("Clean", None, None),
        ("Additive noisy", "additive_gaussian", None),
        ("Additive / MSE", "additive_gaussian", "mse"),
        ("Additive / L1", "additive_gaussian", "l1"),
        ("Gamma noisy", "multiplicative_gamma", None),
        ("Gamma / MSE", "multiplicative_gamma", "mse"),
        ("Gamma / L1", "multiplicative_gamma", "l1"),
    ]
    fig, axes = plt.subplots(count, len(column_specs), figsize=(14, 2.1 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)
    for row in range(count):
        for col, (title, noise_mode, loss_name) in enumerate(column_specs):
            if noise_mode is None:
                image = clean_test[row, 0]
            elif loss_name is None:
                image = noisy_by_mode[noise_mode][row, 0]
            else:
                image = reconstructions[(noise_mode, loss_name)][row, 0]
            axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=9)
    fig.suptitle("Denoising results on held-out grayscale digit images", y=1.002)
    fig.tight_layout()
    fig.savefig(figure_dir / "reconstruction_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot MAE and PSNR in separate axes for readability.
    result_path = output_dir / "metrics.csv"
    with result_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = [f"{r['noise'].replace('_', ' ')}\n{r['loss'].upper()}" for r in rows]
    maes = [float(r["denoised_mae"]) for r in rows]
    psnrs = [float(r["denoised_psnr_db"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(range(len(labels)), maes)
    axes[0].set_title("Held-out MAE (lower is better)")
    axes[0].set_ylabel("MAE")
    axes[1].bar(range(len(labels)), psnrs)
    axes[1].set_title("Held-out PSNR (higher is better)")
    axes[1].set_ylabel("dB")
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels, rotation=15, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "metric_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for noise_mode in NOISE_MODES:
        for loss_name in LOSSES:
            selected = [h for h in histories if h["noise"] == noise_mode and h["loss"] == loss_name]
            axes[0 if noise_mode == "additive_gaussian" else 1].plot(
                [h["epoch"] for h in selected],
                [h["train_loss"] for h in selected],
                marker="o",
                label=f"{loss_name.upper()} loss",
            )
    axes[0].set_title("Additive Gaussian noise")
    axes[1].set_title("Multiplicative Gamma noise")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss (objective-specific units)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.additive_sigma < 0:
        raise SystemExit("--additive-sigma cannot be negative")
    if args.gamma_looks <= 0:
        raise SystemExit("--gamma-looks must be greater than zero")

    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    splits = load_image_splits(
        dataset=args.dataset,
        data_dir=root / args.data_dir,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        seed=args.seed,
    )
    train_images = splits.train
    clean_test = splits.test.cpu()
    print(f"Dataset: {splits.description}")
    print(f"Images: {len(train_images)} train / {len(clean_test)} held-out test")

    # Generate a single held-out noisy tensor per noise model, reused by MSE/L1 runs.
    noisy_by_mode = {}
    for offset, noise_mode in enumerate(NOISE_MODES):
        set_seed(args.seed + 10_000 + offset)
        noisy_by_mode[noise_mode] = add_noise(
            clean_test,
            noise_mode,
            additive_sigma=args.additive_sigma,
            gamma_looks=args.gamma_looks,
        )

    metrics_rows = []
    all_histories = []
    reconstructions = {}
    for noise_mode in NOISE_MODES:
        for loss_name in LOSSES:
            model, history, runtime_seconds, model_path = fit_model(
                train_images=train_images,
                noise_mode=noise_mode,
                loss_name=loss_name,
                args=args,
                device=device,
                model_dir=model_dir,
            )
            all_histories.extend(history)
            baseline, denoised_metrics, denoised = score_condition(
                model, clean_test, noisy_by_mode[noise_mode], device
            )
            reconstructions[(noise_mode, loss_name)] = denoised
            metrics_rows.append({
                "dataset": args.dataset,
                "noise": noise_mode,
                "loss": loss_name,
                "seed": args.seed,
                "epochs": args.epochs,
                "train_images": len(train_images),
                "test_images": len(clean_test),
                "noisy_mse": baseline["mse"],
                "noisy_mae": baseline["mae"],
                "noisy_psnr_db": baseline["psnr_db"],
                "denoised_mse": denoised_metrics["mse"],
                "denoised_mae": denoised_metrics["mae"],
                "denoised_psnr_db": denoised_metrics["psnr_db"],
                "runtime_seconds": round(runtime_seconds, 3),
                "checkpoint": str(Path("models") / Path(model_path).name),
            })
            print(
                f"  held-out MAE {baseline['mae']:.4f} -> {denoised_metrics['mae']:.4f}; "
                f"PSNR {baseline['psnr_db']:.2f} -> {denoised_metrics['psnr_db']:.2f} dB"
            )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("noise", "loss", "epoch", "train_loss"))
        writer.writeheader()
        writer.writerows(all_histories)
    metadata = {
        "dataset": args.dataset,
        "dataset_description": splits.description,
        "train_images": len(train_images),
        "test_images": len(clean_test),
        "image_shape": list(clean_test.shape[1:]),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "additive_sigma": args.additive_sigma,
        "gamma_looks": args.gamma_looks,
        "device": str(device),
        "software": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "numpy": str(np.__version__),
        },
        "conditions": [{"noise": n, "loss": l} for n in NOISE_MODES for l in LOSSES],
        "notes": [
            "Gamma multiplicative noise is a simplified speckle-inspired perturbation, not a physical SAR simulator.",
            "The same held-out noisy inputs are reused for the MSE and L1 model comparison within each noise condition.",
            "The evaluation subset and number of epochs are small; results are illustrative, not generalizable claims.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_figures(clean_test, noisy_by_mode, reconstructions, all_histories, output_dir)
    print(f"\nSaved metrics, plots, metadata, and checkpoints to: {output_dir}")


if __name__ == "__main__":
    main()
