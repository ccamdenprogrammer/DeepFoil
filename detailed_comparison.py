"""
detailed_comparison.py

Side-by-side visual comparison of original vs reconstructed airfoils
with detailed metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.airfoil_vae import AirfoilVAE
from src.data.create_dataset import AirfoilDataset


def main():
    print("="*80)
    print("DETAILED VISUAL COMPARISON")
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

    # Create comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    for i in range(9):
        idx = i * 50  # Space them out
        ax = axes[i // 3, i % 3]

        # Get original
        original = dataset[idx].unsqueeze(0).to(device)

        # Reconstruct
        with torch.no_grad():
            reconstructed, _, _ = model(original)

        # Convert to coords
        orig_coords = original.cpu().numpy().reshape(-1, 2)
        recon_coords = reconstructed.cpu().numpy().reshape(-1, 2)

        # Calculate detailed metrics
        mse = np.mean((orig_coords - recon_coords) ** 2)

        # Y-curvature only (what the model optimizes)
        orig_d2y = np.diff(np.diff(orig_coords[:, 1]))
        recon_d2y = np.diff(np.diff(recon_coords[:, 1]))
        orig_y_curv = np.mean(np.abs(orig_d2y))
        recon_y_curv = np.mean(np.abs(recon_d2y))

        # Plot
        ax.plot(orig_coords[:, 0], orig_coords[:, 1], 'b-',
               linewidth=2, label='Original', alpha=0.7)
        ax.plot(recon_coords[:, 0], recon_coords[:, 1], 'r--',
               linewidth=2, label='Reconstructed')

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x/c', fontsize=8)
        ax.set_ylabel('y/c', fontsize=8)
        ax.set_title(f'Airfoil {idx}\nMSE: {mse:.6f}\n' +
                    f'Y-Curv: {orig_y_curv:.6f}→{recon_y_curv:.6f}',
                    fontsize=9)
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('outputs/plots/detailed_comparison.png', dpi=200)
    print("✅ Saved detailed comparison to outputs/plots/detailed_comparison.png")

    # Print summary statistics
    print("\n"+"="*80)
    print("SUMMARY STATISTICS (50 samples)")
    print("="*80)

    all_mse = []
    all_orig_y_curv = []
    all_recon_y_curv = []

    with torch.no_grad():
        for i in range(50):
            original = dataset[i*10].unsqueeze(0).to(device)
            reconstructed, _, _ = model(original)

            orig_coords = original.cpu().numpy().reshape(-1, 2)
            recon_coords = reconstructed.cpu().numpy().reshape(-1, 2)

            mse = np.mean((orig_coords - recon_coords) ** 2)
            orig_d2y = np.diff(np.diff(orig_coords[:, 1]))
            recon_d2y = np.diff(np.diff(recon_coords[:, 1]))
            orig_y_curv = np.mean(np.abs(orig_d2y))
            recon_y_curv = np.mean(np.abs(recon_d2y))

            all_mse.append(mse)
            all_orig_y_curv.append(orig_y_curv)
            all_recon_y_curv.append(recon_y_curv)

    print(f"\nReconstruction MSE:")
    print(f"  Mean: {np.mean(all_mse):.8f}")
    print(f"  Median: {np.median(all_mse):.8f}")
    print(f"  Max: {np.max(all_mse):.8f}")

    print(f"\nY-Curvature (smoothness):")
    print(f"  Original  - Mean: {np.mean(all_orig_y_curv):.8f}")
    print(f"  Reconstructed - Mean: {np.mean(all_recon_y_curv):.8f}")
    print(f"  Ratio (recon/orig): {np.mean(all_recon_y_curv)/np.mean(all_orig_y_curv):.2f}x")

    ratio = np.mean(all_recon_y_curv)/np.mean(all_orig_y_curv)
    if ratio < 1.5:
        print(f"\n✅ EXCELLENT: Reconstructions are nearly as smooth as originals!")
    elif ratio < 3.0:
        print(f"\n✅ GOOD: Reconstructions are reasonably smooth")
    elif ratio < 5.0:
        print(f"\n⚠️  FAIR: Reconstructions could be smoother")
    else:
        print(f"\n❌ POOR: Reconstructions are significantly less smooth")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
