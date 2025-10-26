"""
train_ae.py

Training script for deterministic Autoencoder.
Simple, no VAE complications, no posterior collapse.
"""

import os
import sys
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.airfoil_ae import AirfoilAE, ae_loss_function
from src.data.create_dataset import AirfoilDataset


def main():
    print("=" * 60)
    print("Training Airfoil Autoencoder (Deterministic)")
    print("=" * 60)

    # Configuration - OPTIMIZED for best quality
    LATENT_DIM = 24           # Increased for more expressive power
    BATCH_SIZE = 32           # Smaller batch for better gradient estimates
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 300          # Much longer training
    WEIGHT_DECAY = 1e-5       # Light regularization

    # Loss weights - will adjust dynamically
    SMOOTHNESS_START = 0.0
    SMOOTHNESS_END = 2.0      # Higher final smoothness
    SMOOTHNESS_RAMP_EPOCHS = 150  # Slower ramp over half training

    DIVERSITY_WEIGHT = 0.0  # NO diversity loss - it causes noise

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load data
    print("\n1) Loading dataset...")
    dataset = AirfoilDataset.load("data/processed/airfoil_dataset.pkl")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")

    # Create model
    print("\n2) Creating model...")
    model = AirfoilAE(input_dim=400, latent_dim=LATENT_DIM, encoder_dropout=0.02)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    print(f"   Latent dim: {LATENT_DIM}")
    print(f"   Architecture: Deterministic Autoencoder (no VAE)")

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler - more aggressive
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )

    # Training history
    train_losses = []
    val_losses = []
    recon_losses = []
    smooth_losses = []
    diversity_losses = []

    best_val_loss = float('inf')

    print("\n3) Training...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Weight decay: {WEIGHT_DECAY}")
    print(f"   Smoothness: {SMOOTHNESS_START} → {SMOOTHNESS_END} over {SMOOTHNESS_RAMP_EPOCHS} epochs")
    print(f"   Diversity weight: {DIVERSITY_WEIGHT}")
    print()

    for epoch in range(1, NUM_EPOCHS + 1):
        # Gradually increase smoothness weight
        if epoch <= SMOOTHNESS_RAMP_EPOCHS:
            smoothness_weight = SMOOTHNESS_START + (SMOOTHNESS_END - SMOOTHNESS_START) * (epoch / SMOOTHNESS_RAMP_EPOCHS)
        else:
            smoothness_weight = SMOOTHNESS_END

        # Train
        model.train()
        tot, trecon, tsmooth, tdiv = 0.0, 0.0, 0.0, 0.0

        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()

            recon, z = model(data)
            loss, recon_loss, smooth_loss, div_loss = ae_loss_function(
                recon, data, z,
                smoothness_weight=smoothness_weight,
                diversity_weight=DIVERSITY_WEIGHT,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot += loss.item()
            trecon += recon_loss.item()
            tsmooth += smooth_loss.item()
            tdiv += div_loss.item()

        n = len(train_loader)
        avg_loss = tot / n
        avg_recon = trecon / n
        avg_smooth = tsmooth / n
        avg_div = tdiv / n

        train_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        smooth_losses.append(avg_smooth)
        diversity_losses.append(avg_div)

        # Validate
        model.eval()
        val_tot = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon, z = model(data)
                loss, _, _, _ = ae_loss_function(
                    recon, data, z,
                    smoothness_weight=smoothness_weight,
                    diversity_weight=DIVERSITY_WEIGHT,
                )
                val_tot += loss.item()

        val_loss = val_tot / len(val_loader)
        val_losses.append(val_loss)

        # Scheduler step
        scheduler.step(val_loss)

        # Progress
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                f"Train: {avg_loss:.6f} | Val: {val_loss:.6f} | "
                f"Recon: {avg_recon:.6f} | Smooth: {avg_smooth:.6f} | "
                f"Div: {avg_div:.6f} | SmoothW: {smoothness_weight:.2f}"
            )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models/airfoil_ae", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'latent_dim': LATENT_DIM,
                'timestamp': datetime.now().isoformat(timespec="seconds"),
            }, "models/airfoil_ae/best_model.pth")

            if epoch % 10 == 0:
                print(f"   → New best model saved! (Val loss: {val_loss:.6f})")

    print(f"\n✅ Training complete!")
    print(f"   Best validation loss: {best_val_loss:.6f}")

    # Plot history
    print("\n4) Plotting training history...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train", linewidth=2)
    axes[0].plot(val_losses, label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(recon_losses, label="Reconstruction", linewidth=2)
    axes[1].plot(smooth_losses, label="Smoothness", linewidth=2)
    axes[1].plot(diversity_losses, label="Diversity", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Components")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/ae_training_history.png", dpi=150)
    print("   Saved to outputs/plots/ae_training_history.png")

    # Save final
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'finished_at': datetime.now().isoformat(timespec="seconds"),
    }, "models/airfoil_ae/final_model.pth")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  ✓ Deterministic Autoencoder (no VAE collapse issues)")
    print(f"  ✓ Latent dimension: {LATENT_DIM}D")
    print(f"  ✓ Best val loss: {best_val_loss:.6f}")
    print("\nModels saved:")
    print("  - models/airfoil_ae/best_model.pth")
    print("  - models/airfoil_ae/final_model.pth")
    print("\nNext: Run visualize_ae.py to check reconstructions")


if __name__ == "__main__":
    main()
