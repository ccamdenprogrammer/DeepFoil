"""
visualize_results.py

Quick visualization and analysis script for trained VAE model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.airfoil_vae import AirfoilVAE
from src.data.create_dataset import AirfoilDataset


def calculate_curvature(coords):
    """Calculate mean absolute curvature (2nd derivative of y)."""
    y = coords[:, 1]
    dy = y[1:] - y[:-1]
    d2y = dy[1:] - dy[:-1]
    return np.mean(np.abs(d2y))


def check_trailing_edge_closure(coords, threshold=0.05):
    """Check if trailing edge is closed."""
    first = coords[0]
    last = coords[-1]
    distance = np.linalg.norm(first - last)
    return distance < threshold, distance


def calculate_diversity(samples):
    """Calculate mean pairwise difference between samples."""
    n = len(samples)
    if n < 2:
        return 0.0

    differences = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.mean(np.abs(samples[i] - samples[j]))
            differences.append(diff)

    return np.mean(differences)


def main():
    print("=" * 60)
    print("Visualizing VAE Results")
    print("=" * 60)

    # Load model
    print("\n1) Loading trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AirfoilVAE(input_dim=400, latent_dim=32, encoder_dropout=0.3)

    checkpoint = torch.load("models/airfoil_vae/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"   ✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Best val loss: {checkpoint.get('val_loss', 'N/A')}")

    # Load dataset
    print("\n2) Loading dataset...")
    dataset = AirfoilDataset.load("data/processed/airfoil_dataset.pkl")

    # Test reconstructions
    print("\n3) Testing reconstructions...")
    n_test = 6
    test_indices = np.random.choice(len(dataset), n_test, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    reconstruction_errors = []

    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            # Get original
            original = dataset[idx].unsqueeze(0).to(device)

            # Reconstruct
            recon, mu, logvar = model(original)

            # Calculate error
            error = torch.mean((recon - original) ** 2).item()
            reconstruction_errors.append(error)

            # Reshape for plotting
            orig_coords = original.cpu().numpy().reshape(-1, 2)
            recon_coords = recon.cpu().numpy().reshape(-1, 2)

            # Plot
            ax = axes[i]
            ax.plot(orig_coords[:, 0], orig_coords[:, 1], 'b-',
                   linewidth=2, label='Original', alpha=0.7)
            ax.plot(recon_coords[:, 0], recon_coords[:, 1], 'r--',
                   linewidth=2, label='Reconstructed')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x/c')
            ax.set_ylabel('y/c')
            ax.set_title(f'Reconstruction {i+1}\nMSE: {error:.6f}', fontsize=10)
            if i == 0:
                ax.legend()

    plt.tight_layout()
    plt.savefig('outputs/plots/reconstructions.png', dpi=150)
    print(f"   ✅ Saved reconstructions to outputs/plots/reconstructions.png")
    print(f"   Mean reconstruction MSE: {np.mean(reconstruction_errors):.6f}")

    # Generate new samples
    print("\n4) Generating new airfoils...")
    n_generate = 10

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    generated_coords_list = []
    curvatures = []
    te_closures = []

    with torch.no_grad():
        generated = model.generate(num_samples=n_generate, device=device)

        for i in range(n_generate):
            coords = generated[i].cpu().numpy().reshape(-1, 2)
            generated_coords_list.append(coords)

            # Calculate metrics
            curv = calculate_curvature(coords)
            is_closed, te_dist = check_trailing_edge_closure(coords)
            curvatures.append(curv)
            te_closures.append(te_dist)

            # Plot
            ax = axes[i]
            ax.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x/c')
            ax.set_ylabel('y/c')
            ax.set_title(f'Generated {i+1}\nTE: {te_dist:.4f}', fontsize=10)

    plt.tight_layout()
    plt.savefig('outputs/plots/generated_airfoils.png', dpi=150)
    print(f"   ✅ Saved generated airfoils to outputs/plots/generated_airfoils.png")

    # Calculate diversity
    diversity = calculate_diversity(np.array(generated_coords_list))

    # Print quality metrics
    print("\n5) Quality Metrics:")
    print(f"   Mean curvature (smoothness): {np.mean(curvatures):.8f}")
    print(f"   Mean TE closure distance:    {np.mean(te_closures):.6f}")
    print(f"   TE closure rate (<0.05):     {np.sum(np.array(te_closures) < 0.05)}/{n_generate}")
    print(f"   Generation diversity:        {diversity:.6f}")

    # Save generated airfoils as CSV
    print("\n6) Saving generated airfoils as CSV...")
    import os
    os.makedirs("outputs/generated_airfoils", exist_ok=True)

    for i, coords in enumerate(generated_coords_list):
        filepath = f"outputs/generated_airfoils/airfoil_{i+1:02d}.csv"
        np.savetxt(filepath, coords, delimiter=',', header='x,y', comments='')

    print(f"   ✅ Saved {n_generate} airfoils to outputs/generated_airfoils/")

    # Final summary
    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")
    print("=" * 60)
    print(f"\nReconstruction MSE: {np.mean(reconstruction_errors):.6f}")
    print(f"Smoothness (curvature): {np.mean(curvatures):.8f}")
    print(f"TE Closure: {np.sum(np.array(te_closures) < 0.05)}/{n_generate} airfoils")
    print(f"Diversity: {diversity:.6f}")
    print(f"\nVisualization files:")
    print(f"  - outputs/plots/reconstructions.png")
    print(f"  - outputs/plots/generated_airfoils.png")
    print(f"  - outputs/plots/training_history.png")
    print(f"  - outputs/generated_airfoils/*.csv")

    # Check for issues
    if np.mean(curvatures) > 0.01:
        print("\n⚠️  Warning: Curvature is high - airfoils may not be smooth")
    else:
        print("\n✅ Airfoils are smooth!")

    if np.sum(np.array(te_closures) < 0.05) < n_generate * 0.9:
        print("⚠️  Warning: Some airfoils have open trailing edges")
    else:
        print("✅ Trailing edges are properly closed!")

    if diversity < 0.001:
        print("⚠️  Warning: Low diversity - latent space may have collapsed")
    else:
        print("✅ Generated airfoils are diverse!")


if __name__ == "__main__":
    main()
