"""
train_vae.py

Training script for Airfoil VAE (Fixed / Proven architecture).

- Encoder: LayerNorm + SiLU
- Decoder: MLP + SiLU (no normalization, no dropout)
- Loss terms averaged (scale-invariant)
- Smoothness regularization on y only (normalized)
- Optional x-monotonicity penalty
- KL β-annealing
"""

import os
import sys
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Make sure src/ is importable if running from scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.airfoil_vae import AirfoilVAE, vae_loss_function
from src.data.create_dataset import AirfoilDataset


class VAETrainer:
    """
    Trainer class for Airfoil VAE.
    """

    def __init__(
        self,
        model: AirfoilVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        smoothness_weight: float = 0.1,
        monotonicity_weight: float = 0.0,
        free_bits: float = 0.0,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm

        # Regularization weights
        self.smoothness_weight = smoothness_weight
        self.monotonicity_weight = monotonicity_weight
        self.free_bits = free_bits

        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []
        self.train_smooth_losses = []
        self.train_mono_losses = []
        self.beta_history = []

        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int, beta: float = 1.0):
        """
        Train for one epoch.

        Returns average total, recon, kl, smooth, mono losses.
        """
        self.model.train()

        tot, trecon, tkl, tsmooth, tmono = 0.0, 0.0, 0.0, 0.0, 0.0

        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)

            # Add input noise to force latent usage (denoising VAE)
            noise_std = 0.01
            noisy_data = data + torch.randn_like(data) * noise_std

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                recon, mu, logvar = self.model(noisy_data)
                loss, recon_loss, kl_loss, smooth_loss, mono_loss = vae_loss_function(
                    recon,
                    data,
                    mu,
                    logvar,
                    beta=beta,
                    smoothness_weight=self.smoothness_weight,
                    monotonicity_weight=self.monotonicity_weight,
                    free_bits=self.free_bits,
                )

            # Backward
            self.scaler.scale(loss).backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track
            tot += loss.item()
            trecon += recon_loss.item()
            tkl += kl_loss.item()
            tsmooth += smooth_loss.item()
            tmono += mono_loss.item()

        n = len(self.train_loader)
        avg_loss = tot / n
        avg_recon = trecon / n
        avg_kl = tkl / n
        avg_smooth = tsmooth / n
        avg_mono = tmono / n

        self.train_losses.append(avg_loss)
        self.train_recon_losses.append(avg_recon)
        self.train_kl_losses.append(avg_kl)
        self.train_smooth_losses.append(avg_smooth)
        self.train_mono_losses.append(avg_mono)
        self.beta_history.append(beta)

        return avg_loss, avg_recon, avg_kl, avg_smooth, avg_mono

    @torch.no_grad()
    def validate(self, beta: float = 1.0):
        """
        Validate the model; returns average validation total loss.
        """
        self.model.eval()

        total = 0.0
        for data in self.val_loader:
            data = data.to(self.device)
            recon, mu, logvar = self.model(data)
            loss, _, _, _, _ = vae_loss_function(
                recon,
                data,
                mu,
                logvar,
                beta=beta,
                smoothness_weight=self.smoothness_weight,
                monotonicity_weight=self.monotonicity_weight,
                free_bits=self.free_bits,
            )
            total += loss.item()

        avg = total / len(self.val_loader)
        self.val_losses.append(avg)
        return avg

    def save_checkpoint(self, epoch: int, filepath: str, extra: dict | None = None):
        """
        Save model checkpoint.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "beta_history": self.beta_history,
            "train_recon_losses": self.train_recon_losses,
            "train_kl_losses": self.train_kl_losses,
            "train_smooth_losses": self.train_smooth_losses,
            "train_mono_losses": self.train_mono_losses,
            "smoothness_weight": self.smoothness_weight,
            "monotonicity_weight": self.monotonicity_weight,
            "free_bits": self.free_bits,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, filepath)

    def plot_history(self, save_path: str = "outputs/plots/training_history.png"):
        """
        Plot training history.
        """
        import numpy as np

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Total loss
        axes[0, 0].plot(self.train_losses, label="Train", linewidth=2)
        axes[0, 0].plot(self.val_losses, label="Val", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss components (train)
        axes[0, 1].plot(self.train_recon_losses, label="Recon (MSE)", linewidth=2)
        axes[0, 1].plot(self.train_kl_losses, label="KL", linewidth=2)
        axes[0, 1].plot(self.train_smooth_losses, label="Smooth (y)", linewidth=2)
        if np.max(self.train_mono_losses) > 0:
            axes[0, 1].plot(self.train_mono_losses, label="Monotonicity (x)", linewidth=2)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Loss Components (Train)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale("log")

        # Beta schedule
        axes[1, 0].plot(self.beta_history, linewidth=2)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Beta")
        axes[1, 0].set_title("Beta Annealing Schedule")
        axes[1, 0].grid(True, alpha=0.3)

        # KL vs Beta
        axes[1, 1].scatter(self.beta_history, self.train_kl_losses, alpha=0.6)
        axes[1, 1].set_xlabel("Beta")
        axes[1, 1].set_ylabel("KL Divergence")
        axes[1, 1].set_title("KL vs Beta (Train)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale("log")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"   Saved training history to {save_path}")


def main():
    """
    Main training function.
    """
    print("=" * 60)
    print("Training Airfoil VAE — Fixed/Proven Architecture")
    print("=" * 60)

    # ---------------- Configuration ----------------
    LATENT_DIM = 32
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50  # Reduced for quick test (original: 200)

    # KL β-annealing - aggressive from the start
    BETA_START = 5.0  # Very aggressive start
    BETA_END = 10.0   # Very high beta
    BETA_ANNEAL_EPOCHS = 25  # Reduced proportionally (original: 100)

    # Regularization weights from the new loss
    SMOOTHNESS_WEIGHT = 0.01       # Reduce to allow more flexibility
    MONOTONICITY_WEIGHT = 0.0      # set >0 to discourage x backtracking
    FREE_BITS = 0.0                # Remove free bits - didn't help

    # Trainer options
    USE_AMP = torch.cuda.is_available()
    MAX_GRAD_NORM = 1.0

    # ---------------- Device ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ---------------- Data ----------------
    print("\n1) Loading dataset...")
    dataset = AirfoilDataset.load("data/processed/airfoil_dataset.pkl")

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")

    # ---------------- Model ----------------
    print("\n2) Creating model...")
    model = AirfoilVAE(input_dim=400, latent_dim=LATENT_DIM, encoder_dropout=0.3)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters:       {num_params:,}")
    print(f"   Latent dimension: {LATENT_DIM}")
    print(f"   Encoder: LayerNorm + SiLU + Dropout(0.3)  |  Decoder: MLP + SiLU")

    # ---------------- Trainer ----------------
    trainer = VAETrainer(
        model,
        train_loader,
        val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        smoothness_weight=SMOOTHNESS_WEIGHT,
        monotonicity_weight=MONOTONICITY_WEIGHT,
        free_bits=FREE_BITS,
        max_grad_norm=MAX_GRAD_NORM,
        use_amp=USE_AMP,
    )

    # ---------------- Training Loop ----------------
    print("\n3) Training...")
    print(f"   Epochs:           {NUM_EPOCHS}")
    print(f"   Learning rate:    {LEARNING_RATE}")
    print(f"   β-anneal:         {BETA_START} → {BETA_END} over {BETA_ANNEAL_EPOCHS} epochs")
    print(f"   Free bits:        {FREE_BITS}")
    print(f"   Smoothness (y):   {SMOOTHNESS_WEIGHT}")
    print(f"   Monotonicity (x): {MONOTONICITY_WEIGHT}")
    print()

    for epoch in range(1, NUM_EPOCHS + 1):
        # Linear β-anneal
        if epoch <= BETA_ANNEAL_EPOCHS:
            beta = BETA_START + (BETA_END - BETA_START) * (epoch / BETA_ANNEAL_EPOCHS)
        else:
            beta = BETA_END

        # Train
        train_loss, recon_loss, kl_loss, smooth_loss, mono_loss = trainer.train_epoch(epoch, beta=beta)

        # Validate
        val_loss = trainer.validate(beta=beta)

        # Progress
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Recon: {recon_loss:.4f} | KL: {kl_loss:.4f} | "
                f"Smooth: {smooth_loss:.4f} | Mono: {mono_loss:.6f} | "
                f"Beta: {beta:.3f}"
            )

        # Save best (after warmup)
        if epoch > BETA_ANNEAL_EPOCHS and val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.save_checkpoint(
                epoch,
                "models/airfoil_vae/best_model.pth",
                extra={
                    "val_loss": val_loss,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            )
            if epoch % 10 == 0:
                print(f"   → New best model saved! (Val loss: {val_loss:.4f})")

        # Periodic checkpoint
        if epoch % 50 == 0:
            trainer.save_checkpoint(
                epoch,
                f"models/airfoil_vae/checkpoint_epoch_{epoch}.pth",
                extra={"val_loss": val_loss},
            )

    print(f"\n✅ Training complete!")
    print(f"   Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"   Final KL divergence : {trainer.train_kl_losses[-1]:.4f}")

    # ---------------- Post-training ----------------
    print("\n4) Plotting training history...")
    trainer.plot_history()

    # Save final
    trainer.save_checkpoint(
        NUM_EPOCHS,
        "models/airfoil_vae/final_model.pth",
        extra={
            "best_val_loss": trainer.best_val_loss,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    # Quick sanity check for KL collapse
    final_kl = trainer.train_kl_losses[-1]
    if final_kl < 1.0:
        print("\n⚠️  WARNING: KL divergence is low!")
        print(f"   Final KL: {final_kl:.4f}")
        print("   The latent space might have collapsed.")
        print("   Consider: increase BETA_END, reduce reconstruction pressure,")
        print("             or extend training.")
    else:
        print(f"\n✅ KL divergence looks healthy: {final_kl:.4f}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nArchitecture summary:")
    print("  ✓ Encoder: LayerNorm + SiLU")
    print("  ✓ Decoder: MLP + SiLU (no normalization, no dropout)")
    print("  ✓ Averaged losses; y-only smoothness; optional x-monotonicity")
    print(f"  ✓ Latent space: {LATENT_DIM}D")
    print("\nModels saved:")
    print("  - models/airfoil_vae/best_model.pth")
    print("  - models/airfoil_vae/final_model.pth")
    print("\nNext: run visualize_vae.py to inspect generations.")


if __name__ == "__main__":
    main()
