"""
visualize_ae.py

Visualization and analysis for trained Autoencoder.
Includes generation via sampling from learned latent distribution.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.airfoil_ae import AirfoilAE
from src.data.create_dataset import AirfoilDataset


def main():
    print("="*80)
    print("AIRFOIL AUTOENCODER VISUALIZATION")
    print("="*80)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("models/airfoil_ae/best_model.pth", map_location=device)
    latent_dim = checkpoint['latent_dim']

    model = AirfoilAE(input_dim=400, latent_dim=latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"\n✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Latent dimension: {latent_dim}")
    print(f"   Val loss: {checkpoint['val_loss']:.6f}")

    # Load dataset
    dataset = AirfoilDataset.load("data/processed/airfoil_dataset.pkl")

    # 1) Collect latent codes from entire dataset
    print("\n1) Collecting latent codes...")
    all_latents = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i].unsqueeze(0).to(device)
            z = model.encode(data)
            all_latents.append(z.cpu().numpy())

    all_latents = np.array(all_latents).squeeze()  # (N, latent_dim)

    print(f"   Collected {len(all_latents)} latent codes")
    print(f"   Latent mean: {np.mean(all_latents, axis=0)[:5]}...")
    print(f"   Latent std:  {np.std(all_latents, axis=0)[:5]}...")

    # Check diversity
    latent_std = np.std(all_latents, axis=0)
    print(f"\n   Latent diversity check:")
    print(f"   Mean std per dimension: {np.mean(latent_std):.6f}")
    print(f"   Min std: {np.min(latent_std):.6f}")
    print(f"   Max std: {np.max(latent_std):.6f}")

    if np.mean(latent_std) < 0.01:
        print("   ⚠️  WARNING: Latent codes have low diversity")
    else:
        print("   ✅ Latent codes are diverse")

    # 2) Fit simple Gaussian distribution for generation
    print("\n2) Fitting Gaussian distribution to latent space...")
    latent_mean = np.mean(all_latents, axis=0)
    latent_cov = np.cov(all_latents.T)

    print(f"   Fitted Gaussian distribution")
    print(f"   Mean: {latent_mean[:3]}...")
    print(f"   Cov diagonal: {np.diag(latent_cov)[:3]}...")

    # 3) Test reconstructions
    print("\n3) Generating reconstruction visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    with torch.no_grad():
        for i in range(6):
            idx = i * 250
            ax = axes[i // 3, i % 3]

            original = dataset[idx].unsqueeze(0).to(device)
            recon, z = model(original)

            orig_coords = original.cpu().numpy().reshape(-1, 2)
            recon_coords = recon.cpu().numpy().reshape(-1, 2)

            mse = np.mean((orig_coords - recon_coords) ** 2)

            ax.plot(orig_coords[:, 0], orig_coords[:, 1], 'b-',
                   linewidth=2, label='Original', alpha=0.7)
            ax.plot(recon_coords[:, 0], recon_coords[:, 1], 'r--',
                   linewidth=2, label='Reconstructed')

            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x/c', fontsize=8)
            ax.set_ylabel('y/c', fontsize=8)
            ax.set_title(f'Reconstruction {i+1}\nMSE: {mse:.6f}', fontsize=9)
            if i == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('outputs/plots/ae_reconstructions.png', dpi=200)
    print("   ✅ Saved to outputs/plots/ae_reconstructions.png")

    # Calculate average reconstruction error
    print("\n4) Computing reconstruction metrics...")
    recon_errors = []
    with torch.no_grad():
        for i in range(min(100, len(dataset))):
            original = dataset[i].unsqueeze(0).to(device)
            recon, _ = model(original)
            mse = torch.mean((recon - original) ** 2).item()
            recon_errors.append(mse)

    print(f"   Mean reconstruction MSE: {np.mean(recon_errors):.8f}")
    print(f"   Median reconstruction MSE: {np.median(recon_errors):.8f}")
    print(f"   Max reconstruction MSE: {np.max(recon_errors):.8f}")

    # 5) Generate new airfoils by sampling from Gaussian
    print("\n5) Generating new airfoils via Gaussian sampling...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 6))

    with torch.no_grad():
        for i in range(10):
            ax = axes[i // 5, i % 5]

            # Sample from Gaussian
            z_sample = np.random.multivariate_normal(latent_mean, latent_cov, size=1)
            z_tensor = torch.tensor(z_sample, dtype=torch.float32).to(device)

            # Decode
            generated = model.decode(z_tensor)
            gen_coords = generated.cpu().numpy().reshape(-1, 2)

            # Calculate trailing edge closure
            te_dist = np.linalg.norm(gen_coords[0] - gen_coords[-1])

            ax.plot(gen_coords[:, 0], gen_coords[:, 1], 'g-', linewidth=2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x/c', fontsize=8)
            ax.set_ylabel('y/c', fontsize=8)
            ax.set_title(f'Generated {i+1}\nTE: {te_dist:.4f}', fontsize=9)

    plt.tight_layout()
    plt.savefig('outputs/plots/ae_generated_airfoils.png', dpi=200)
    print("   ✅ Saved to outputs/plots/ae_generated_airfoils.png")

    # Calculate generation quality metrics
    print("\n6) Generation quality metrics...")
    te_closures = []
    smoothness_vals = []

    with torch.no_grad():
        for i in range(50):
            z_sample = np.random.multivariate_normal(latent_mean, latent_cov, size=1)
            z_tensor = torch.tensor(z_sample, dtype=torch.float32).to(device)
            generated = model.decode(z_tensor)
            gen_coords = generated.cpu().numpy().reshape(-1, 2)

            # Trailing edge
            te_dist = np.linalg.norm(gen_coords[0] - gen_coords[-1])
            te_closures.append(te_dist)

            # Smoothness
            d2y = np.diff(np.diff(gen_coords[:, 1]))
            smoothness = np.mean(np.abs(d2y))
            smoothness_vals.append(smoothness)

    print(f"   Mean TE closure: {np.mean(te_closures):.6f}")
    print(f"   TE closure rate (<0.05): {np.sum(np.array(te_closures) < 0.05)}/50")
    print(f"   Mean smoothness (curvature): {np.mean(smoothness_vals):.6f}")

    if np.sum(np.array(te_closures) < 0.05) >= 45:
        print("   ✅ Excellent trailing edge closure")
    if np.mean(smoothness_vals) < 0.005:
        print("   ✅ Generated airfoils are smooth")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Results:")
    print(f"  • Reconstruction MSE: {np.mean(recon_errors):.6f}")
    print(f"  • Latent diversity: {np.mean(latent_std):.6f}")
    print(f"  • Generation TE closure: {np.mean(te_closures):.6f}")
    print(f"  • Generation smoothness: {np.mean(smoothness_vals):.6f}")


if __name__ == "__main__":
    main()
