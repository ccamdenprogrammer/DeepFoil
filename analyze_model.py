"""
analyze_model.py

Detailed quantitative analysis of VAE model performance.
Measures reconstruction quality, smoothness, and other metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.airfoil_vae import AirfoilVAE
from src.data.create_dataset import AirfoilDataset


def calculate_reconstruction_metrics(original, reconstructed):
    """Calculate detailed reconstruction metrics."""
    # Reshape to coordinates
    orig_coords = original.reshape(-1, 2)
    recon_coords = reconstructed.reshape(-1, 2)

    # Point-wise MSE
    mse = np.mean((orig_coords - recon_coords) ** 2)

    # Point-wise MAE
    mae = np.mean(np.abs(orig_coords - recon_coords))

    # Max error
    max_error = np.max(np.abs(orig_coords - recon_coords))

    # Separate x and y errors
    x_error = np.mean((orig_coords[:, 0] - recon_coords[:, 0]) ** 2)
    y_error = np.mean((orig_coords[:, 1] - recon_coords[:, 1]) ** 2)

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'x_mse': x_error,
        'y_mse': y_error
    }


def calculate_smoothness_metrics(coords):
    """Calculate smoothness using various metrics."""
    # First derivative (velocity)
    dx = np.diff(coords[:, 0])
    dy = np.diff(coords[:, 1])

    # Second derivative (acceleration/curvature)
    d2x = np.diff(dx)
    d2y = np.diff(dy)

    # Mean absolute curvature
    mean_curvature_x = np.mean(np.abs(d2x))
    mean_curvature_y = np.mean(np.abs(d2y))

    # Max curvature (detect sharp corners)
    max_curvature_x = np.max(np.abs(d2x))
    max_curvature_y = np.max(np.abs(d2y))

    # Standard deviation of curvature (smoothness consistency)
    std_curvature_x = np.std(d2x)
    std_curvature_y = np.std(d2y)

    # Curvature metric (combined)
    total_curvature = np.sqrt(d2x**2 + d2y**2)
    mean_total_curvature = np.mean(total_curvature)
    max_total_curvature = np.max(total_curvature)

    return {
        'mean_curv_x': mean_curvature_x,
        'mean_curv_y': mean_curvature_y,
        'max_curv_x': max_curvature_x,
        'max_curv_y': max_curvature_y,
        'std_curv_x': std_curvature_x,
        'std_curv_y': std_curvature_y,
        'mean_total_curv': mean_total_curvature,
        'max_total_curv': max_total_curvature
    }


def check_trailing_edge_closure(coords, threshold=0.01):
    """Check if trailing edge is closed."""
    first = coords[0]
    last = coords[-1]
    distance = np.linalg.norm(first - last)
    return distance < threshold, distance


def analyze_latent_space(model, dataset, device, num_samples=100):
    """Analyze latent space properties."""
    latent_means = []
    latent_stds = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            data = dataset[i].unsqueeze(0).to(device)
            mu, logvar = model.encode(data)
            std = torch.exp(0.5 * logvar)

            latent_means.append(mu.cpu().numpy())
            latent_stds.append(std.cpu().numpy())

    latent_means = np.array(latent_means).squeeze()
    latent_stds = np.array(latent_stds).squeeze()

    return {
        'mean_mu': np.mean(latent_means, axis=0),
        'std_mu': np.std(latent_means, axis=0),
        'mean_std': np.mean(latent_stds, axis=0),
        'overall_mean_std': np.mean(latent_stds),
        'latent_usage': np.mean(np.std(latent_means, axis=0))  # How much each dimension varies
    }


def main():
    print("=" * 80)
    print("DETAILED MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Load model
    print("\n1. Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AirfoilVAE(input_dim=400, latent_dim=32, encoder_dropout=0.3)

    try:
        checkpoint = torch.load("models/airfoil_vae/best_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"   ✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    except FileNotFoundError:
        print("   ❌ No trained model found. Please train first.")
        return

    # Load dataset
    print("\n2. Loading dataset...")
    dataset = AirfoilDataset.load("data/processed/airfoil_dataset.pkl")
    print(f"   ✅ Loaded {len(dataset)} airfoils")

    # Analyze reconstruction quality
    print("\n3. Analyzing Reconstruction Quality...")
    print("-" * 80)

    n_test = 50
    all_recon_metrics = []

    with torch.no_grad():
        for i in range(n_test):
            original = dataset[i].unsqueeze(0).to(device)
            reconstructed, mu, logvar = model(original)

            orig_np = original.cpu().numpy().flatten()
            recon_np = reconstructed.cpu().numpy().flatten()

            metrics = calculate_reconstruction_metrics(orig_np, recon_np)
            all_recon_metrics.append(metrics)

    # Aggregate reconstruction metrics
    avg_mse = np.mean([m['mse'] for m in all_recon_metrics])
    avg_mae = np.mean([m['mae'] for m in all_recon_metrics])
    avg_max_error = np.mean([m['max_error'] for m in all_recon_metrics])
    worst_mse = np.max([m['mse'] for m in all_recon_metrics])

    print(f"   Average MSE:        {avg_mse:.8f}")
    print(f"   Average MAE:        {avg_mae:.8f}")
    print(f"   Average Max Error:  {avg_max_error:.6f}")
    print(f"   Worst MSE:          {worst_mse:.8f}")

    # Check if reconstruction is good enough
    if avg_mse > 0.001:
        print(f"   ⚠️  WARNING: MSE is high! Target is < 0.0001")
    else:
        print(f"   ✅ Reconstruction quality is good")

    # Analyze smoothness of reconstructions
    print("\n4. Analyzing Smoothness of Reconstructed Airfoils...")
    print("-" * 80)

    recon_smoothness = []
    original_smoothness = []

    with torch.no_grad():
        for i in range(n_test):
            original = dataset[i].unsqueeze(0).to(device)
            reconstructed, _, _ = model(original)

            orig_coords = original.cpu().numpy().reshape(-1, 2)
            recon_coords = reconstructed.cpu().numpy().reshape(-1, 2)

            orig_smooth = calculate_smoothness_metrics(orig_coords)
            recon_smooth = calculate_smoothness_metrics(recon_coords)

            original_smoothness.append(orig_smooth)
            recon_smoothness.append(recon_smooth)

    # Compare smoothness
    orig_mean_curv = np.mean([s['mean_total_curv'] for s in original_smoothness])
    recon_mean_curv = np.mean([s['mean_total_curv'] for s in recon_smoothness])
    orig_max_curv = np.mean([s['max_total_curv'] for s in original_smoothness])
    recon_max_curv = np.mean([s['max_total_curv'] for s in recon_smoothness])

    print(f"   Original airfoils:")
    print(f"      Mean curvature:  {orig_mean_curv:.8f}")
    print(f"      Max curvature:   {orig_max_curv:.8f}")
    print(f"   Reconstructed airfoils:")
    print(f"      Mean curvature:  {recon_mean_curv:.8f}")
    print(f"      Max curvature:   {recon_max_curv:.8f}")

    smoothness_ratio = recon_mean_curv / orig_mean_curv
    print(f"   Smoothness ratio (recon/orig): {smoothness_ratio:.4f}")

    if smoothness_ratio > 1.5:
        print(f"   ⚠️  WARNING: Reconstructions are LESS smooth than originals!")
    elif smoothness_ratio < 0.5:
        print(f"   ⚠️  WARNING: Reconstructions may be over-smoothed (losing detail)")
    else:
        print(f"   ✅ Smoothness is well-preserved")

    # Analyze generated samples
    print("\n5. Analyzing Generated Samples...")
    print("-" * 80)

    n_generated = 20
    with torch.no_grad():
        generated = model.generate(num_samples=n_generated, device=device)

    gen_smoothness = []
    gen_te_closure = []

    for i in range(n_generated):
        coords = generated[i].cpu().numpy().reshape(-1, 2)
        smooth = calculate_smoothness_metrics(coords)
        is_closed, distance = check_trailing_edge_closure(coords)

        gen_smoothness.append(smooth)
        gen_te_closure.append(distance)

    gen_mean_curv = np.mean([s['mean_total_curv'] for s in gen_smoothness])
    gen_max_curv = np.mean([s['max_total_curv'] for s in gen_smoothness])
    te_closure_rate = np.sum(np.array(gen_te_closure) < 0.01) / n_generated

    print(f"   Mean curvature:     {gen_mean_curv:.8f}")
    print(f"   Max curvature:      {gen_max_curv:.8f}")
    print(f"   TE closure rate:    {te_closure_rate*100:.1f}% (< 0.01 threshold)")
    print(f"   Mean TE distance:   {np.mean(gen_te_closure):.6f}")

    if gen_mean_curv > 0.01:
        print(f"   ⚠️  WARNING: Generated airfoils are NOT smooth!")
    else:
        print(f"   ✅ Generated airfoils are smooth")

    # Analyze latent space
    print("\n6. Analyzing Latent Space...")
    print("-" * 80)

    latent_stats = analyze_latent_space(model, dataset, device)

    print(f"   Mean latent std:       {latent_stats['overall_mean_std']:.8f}")
    print(f"   Latent usage (std):    {latent_stats['latent_usage']:.8f}")

    if latent_stats['overall_mean_std'] < 0.01:
        print(f"   ⚠️  WARNING: Latent variance is very low (posterior collapse!)")
    else:
        print(f"   ✅ Latent space has good variance")

    if latent_stats['latent_usage'] < 0.1:
        print(f"   ⚠️  WARNING: Latent dimensions are not being used effectively")
    else:
        print(f"   ✅ Latent dimensions are well-utilized")

    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)

    issues = []

    if avg_mse > 0.001:
        issues.append("❌ Poor reconstruction quality (high MSE)")

    if smoothness_ratio > 1.5:
        issues.append("❌ Reconstructions are jagged/not smooth")

    if gen_mean_curv > 0.01:
        issues.append("❌ Generated airfoils are not smooth")

    if latent_stats['overall_mean_std'] < 0.01:
        issues.append("❌ Posterior collapse detected")

    if te_closure_rate < 0.9:
        issues.append("❌ Poor trailing edge closure")

    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")

        print("\nRECOMMENDATIONS:")
        if avg_mse > 0.001:
            print("  • Reduce beta (KL weight) to focus more on reconstruction")
            print("  • Train for more epochs")

        if smoothness_ratio > 1.5 or gen_mean_curv > 0.01:
            print("  • Increase smoothness_weight (currently 0.01)")
            print("  • Add more points per airfoil or use higher-order interpolation")

        if latent_stats['overall_mean_std'] < 0.01:
            print("  • Reduce beta to prevent posterior collapse")
            print("  • Increase encoder dropout")
            print("  • Add noise to inputs during training")

        if te_closure_rate < 0.9:
            print("  • Add explicit trailing edge closure loss")
    else:
        print("\n✅ All metrics look good!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
