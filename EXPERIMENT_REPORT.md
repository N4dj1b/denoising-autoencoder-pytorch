# Experiment report: denoising autoencoder with two reconstruction losses

## Objective

Evaluate a small convolutional denoising autoencoder under two simple image corruption processes and compare MSE-trained and L1-trained models using the same held-out noisy examples within each corruption condition.

## Data and protocol

- **Dataset:** scikit-learn Digits, a public dataset containing 1,797 grayscale 8×8 images.
- **Preprocessing:** divide source pixel values by 16, then bilinearly resize images to 28×28 and keep values in [0, 1]. No class labels are used as model targets.
- **Split:** fixed-seed random 80/20 split; the run uses the first 1,200 examples from the training split and 350 from the held-out split.
- **Model:** convolutional encoder (28×28 → 14×14 → 7×7) and decoder (7×7 → 14×14 → 28×28) with ReLU activations and a sigmoid output.
- **Optimization:** Adam, learning rate 0.001, batch size 128, six epochs, seed 42.
- **Conditions:** additive Gaussian noise (sigma 0.25) or multiplicative Gamma factor (shape/rate 4), each trained with MSE or L1 reconstruction loss.
- **Evaluation:** compute MSE, MAE and PSNR on 350 held-out clean images and corresponding corrupted/reconstructed images. Within each corruption type, both losses are evaluated using the exact same noisy test tensor.

## Results

| Corruption | Objective | MSE (input → output) | MAE (input → output) | PSNR (input → output, dB) |
|---|---|---:|---:|---:|
| Additive Gaussian | MSE | 0.04035 → 0.03002 | 0.14361 → 0.11966 | 13.941 → 15.226 |
| Additive Gaussian | L1 | 0.04035 → 0.02411 | 0.14361 → 0.10200 | 13.941 → 16.178 |
| Multiplicative Gamma | MSE | 0.02707 → 0.02596 | 0.09595 → 0.10595 | 15.676 → 15.856 |
| Multiplicative Gamma | L1 | 0.02707 → 0.02355 | 0.09595 → 0.10176 | 15.676 → 16.280 |

The complete machine-readable table is in `results/metrics.csv`. Per-epoch objective values are in `results/training_history.csv`; run parameters are in `results/run_metadata.json`.

## Interpretation

For additive Gaussian noise, both learned reconstructions improve over the corrupted input on MAE and PSNR. In this run, the L1-trained model yields the best MAE (0.1020) and PSNR (16.18 dB). For multiplicative Gamma noise, both models slightly improve PSNR but produce worse MAE than the noisy input. This difference demonstrates why evaluation should use several metrics and more than one corruption type.

The absolute metrics across the two noise conditions are not directly comparable as a measure of difficulty: the effective corruption differs due to the noise distributions, signal dependence and clipping to [0, 1]. These numbers are a single small experiment, not a statistically reliable ranking of MSE versus L1 generally.

## Limitations

1. The source digits are 8×8. Upscaling is for compatibility and visualization, not new information.
2. One seed, short training and small subsets do not support claims about generalization.
3. Gamma multiplicative noise with unit-mean factors is only a simplified speckle-inspired perturbation. It does not model SAR acquisition physics, the temporal dimension of SAR time series, or SAR-specific statistical/geometric effects.
4. The task is supervised denoising (the clean image is the reconstruction target), not a full self-supervised latent-representation learning system.

## Reproduction

```bash
pip install -r requirements.txt
python -m pytest -q
python -m src.train --dataset digits --epochs 6 --train-limit 1200 --test-limit 350 --seed 42
```
