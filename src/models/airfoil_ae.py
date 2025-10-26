"""
airfoil_ae.py

Simple deterministic Autoencoder for airfoil generation.
No VAE, no posterior collapse - just clean reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AirfoilAE(nn.Module):
    """
    Simple Autoencoder for airfoils.

    No variational component - just encoder -> latent -> decoder.
    """

    def __init__(
        self,
        input_dim: int = 400,  # 200 (x,y) pairs
        latent_dim: int = 16,
        encoder_dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: input -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(encoder_dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(encoder_dropout),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(encoder_dropout),

            nn.Linear(64, latent_dim),  # No activation - latent code
        )

        # Decoder: latent -> output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),

            nn.Linear(64, 128),
            nn.SiLU(),

            nn.Linear(128, 256),
            nn.SiLU(),

            nn.Linear(256, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.

        Returns:
            recon: reconstructed input
            z: latent code
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def ae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    *,
    smoothness_weight: float = 1.0,
    diversity_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Autoencoder loss with smoothness and diversity regularization.

    Args:
        recon_x: (B, 2*N)
        x:       (B, 2*N)
        z:       (B, latent_dim) latent codes
        smoothness_weight: weight for y curvature penalty
        diversity_weight: weight for encouraging diverse latent codes

    Returns:
        total_loss, recon_loss, smooth_loss, diversity_loss
    """
    B, D = recon_x.shape

    # 1) Reconstruction
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")

    # 2) Smoothness on y only
    coords = recon_x.view(B, D // 2, 2)  # (B, N, 2)
    y = coords[..., 1]  # (B, N)

    dy = y[:, 1:] - y[:, :-1]
    d2y = dy[:, 1:] - dy[:, :-1]
    smooth_loss = torch.mean(d2y ** 2)

    # 3) Diversity loss: encourage different latent codes for different inputs
    # Maximize variance of latent codes in batch
    z_var = torch.var(z, dim=0)  # (latent_dim,)
    diversity_loss = -torch.mean(z_var)  # Negative variance = penalty

    total = recon_loss + smoothness_weight * smooth_loss + diversity_weight * diversity_loss
    return total, recon_loss, smooth_loss, diversity_loss


if __name__ == "__main__":
    # Quick test
    model = AirfoilAE(input_dim=400, latent_dim=16)
    x = torch.randn(8, 400)
    recon, z = model(x)

    loss, recon_loss, smooth_loss, div_loss = ae_loss_function(recon, x, z)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape:  {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Recon shape:  {recon.shape}")
    print(f"\nLosses:")
    print(f"  Total: {loss.item():.6f}")
    print(f"  Recon: {recon_loss.item():.6f}")
    print(f"  Smooth: {smooth_loss.item():.6f}")
    print(f"  Diversity: {div_loss.item():.6f}")
