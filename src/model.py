"""A small convolutional denoising autoencoder for 28x28 grayscale images."""

from torch import nn


class DenoisingAutoencoder(nn.Module):
    """Map a noisy [B, 1, 28, 28] image to a cleaned [B, 1, 28, 28] image."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # keep predictions in [0, 1]
        )

    def forward(self, noisy_image):
        return self.decoder(self.encoder(noisy_image))
