"""
diagnose_collapse.py

Diagnose posterior collapse by checking if latent codes vary with input.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.airfoil_vae import AirfoilVAE
from src.data.create_dataset import AirfoilDataset


def main():
    print("="*80)
    print("DIAGNOSING POSTERIOR COLLAPSE")
    print("="*80)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AirfoilVAE(input_dim=400, latent_dim=32)
    checkpoint = torch.load("models/airfoil_vae/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load dataset
    dataset = AirfoilDataset.load("data/processed/airfoil_dataset.pkl")

    # Encode 20 different airfoils
    latent_codes = []
    reconstructions = []

    with torch.no_grad():
        for i in range(20):
            original = dataset[i*50].unsqueeze(0).to(device)
            mu, logvar = model.encode(original)
            recon, _, _ = model(original)

            latent_codes.append(mu.cpu().numpy())
            reconstructions.append(recon.cpu().numpy())

    latent_codes = np.array(latent_codes).squeeze()
    reconstructions = np.array(reconstructions).squeeze()

    # Check variance in latent space
    print("\n1) LATENT SPACE ANALYSIS")
    print("-" * 80)
    latent_std = np.std(latent_codes, axis=0)
    latent_mean_std = np.mean(latent_std)

    print(f"Latent code std per dimension (mean): {latent_mean_std:.8f}")
    print(f"Latent code std per dimension (min):  {np.min(latent_std):.8f}")
    print(f"Latent code std per dimension (max):  {np.max(latent_std):.8f}")
    print(f"Number of dimensions with std < 0.01:  {np.sum(latent_std < 0.01)} / 32")
    print(f"Number of dimensions with std < 0.001: {np.sum(latent_std < 0.001)} / 32")

    if latent_mean_std < 0.01:
        print("\n❌ POSTERIOR COLLAPSE DETECTED!")
        print("   All latent codes are nearly identical - encoder is not encoding input info")
    elif np.sum(latent_std < 0.01) > 25:
        print("\n⚠️  SEVERE COLLAPSE: Most dimensions collapsed")
    elif np.sum(latent_std < 0.01) > 15:
        print("\n⚠️  MODERATE COLLAPSE: Many dimensions collapsed")
    else:
        print("\n✅ Latent space looks healthy")

    # Check variance in reconstructions
    print("\n2) RECONSTRUCTION DIVERSITY")
    print("-" * 80)
    recon_std = np.std(reconstructions, axis=0)
    recon_mean_std = np.mean(recon_std)

    print(f"Reconstruction std per coordinate (mean): {recon_mean_std:.8f}")
    print(f"Reconstruction std per coordinate (min):  {np.min(recon_std):.8f}")
    print(f"Reconstruction std per coordinate (max):  {np.max(recon_std):.8f}")

    if recon_mean_std < 0.001:
        print("\n❌ RECONSTRUCTIONS ARE IDENTICAL!")
        print("   Model is outputting the same airfoil regardless of input")
    elif recon_mean_std < 0.005:
        print("\n⚠️  LOW DIVERSITY: Reconstructions are too similar")
    else:
        print("\n✅ Reconstructions show good diversity")

    # Check if mu is close to prior (0,1)
    print("\n3) ENCODER OUTPUT vs PRIOR")
    print("-" * 80)
    mu_mean = np.mean(latent_codes, axis=0)
    mu_global_mean = np.mean(np.abs(mu_mean))

    print(f"Mean of mu across samples: {mu_global_mean:.8f}")
    print(f"Expected (should be near 0 but not exactly): ~0.05-0.5")

    if mu_global_mean < 0.001:
        print("\n❌ ENCODER COLLAPSED TO PRIOR!")
        print("   Encoder is outputting ~N(0,1) regardless of input")
    else:
        print("\n✅ Encoder is encoding information")

    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)

    if latent_mean_std < 0.01 and recon_mean_std < 0.001:
        print("\n❌ SEVERE POSTERIOR COLLAPSE")
        print("\nROOT CAUSE:")
        print("  - KL divergence penalty forced latent codes to collapse to prior N(0,1)")
        print("  - Smoothness weight too high, making decoder ignore latent codes")
        print("  - Beta value too low early in training, then couldn't recover")
        print("\nFIX REQUIRED:")
        print("  1. INCREASE beta_start and beta_end (prevent KL collapse)")
        print("  2. REDUCE smoothness_weight (currently 100.0 - way too high)")
        print("  3. Use free_bits > 0 to maintain minimum KL divergence")
        print("  4. Retrain from scratch")
    else:
        print("\n✅ Model looks healthy")


if __name__ == "__main__":
    main()
