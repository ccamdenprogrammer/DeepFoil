"""
generate_airfoils.py

PRODUCTION-LEVEL airfoil generator building on the working AE.
Samples from the ACTUAL learned latent distribution, not random N(0,1).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.mixture import GaussianMixture

from src.models.airfoil_ae import AirfoilAE
from src.data.create_dataset import AirfoilDataset


class AirfoilGenerator:
    """Production-ready airfoil generator using trained AE."""

    def __init__(self, model_path="models/airfoil_ae/best_model.pth",
                 dataset_path="data/processed/airfoil_dataset.pkl"):
        """Initialize generator with trained model and fit latent distribution."""

        print("Initializing Airfoil Generator...")

        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.latent_dim = checkpoint.get('latent_dim', 24)

        self.model = AirfoilAE(latent_dim=self.latent_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"  Loaded AE model (latent_dim={self.latent_dim})")

        # Load dataset and encode ALL airfoils to learn distribution
        dataset = AirfoilDataset.load(dataset_path)
        print(f"  Encoding {len(dataset)} training airfoils...")

        latent_codes = []
        with torch.no_grad():
            for i in range(len(dataset)):
                airfoil = dataset[i].unsqueeze(0)
                z = self.model.encode(airfoil)
                latent_codes.append(z)

                if (i + 1) % 500 == 0:
                    print(f"    Encoded {i + 1}/{len(dataset)}...")

        self.latent_codes = torch.cat(latent_codes, dim=0).numpy()  # (N, latent_dim)

        print(f"\n  Latent distribution stats:")
        print(f"    Mean: {self.latent_codes.mean():.3f}")
        print(f"    Std:  {self.latent_codes.std():.3f}")
        print(f"    Range: [{self.latent_codes.min():.3f}, {self.latent_codes.max():.3f}]")

        # Fit Gaussian Mixture Model to latent codes for better sampling
        print(f"\n  Fitting Gaussian Mixture Model (GMM) to latent distribution...")
        self.gmm = GaussianMixture(n_components=8, covariance_type='full', random_state=42)
        self.gmm.fit(self.latent_codes)

        print(f"    GMM BIC score: {self.gmm.bic(self.latent_codes):.1f}")
        print(f"\nGenerator ready!")

    def generate(self, num_samples=1, method='gmm', diversity=1.0):
        """
        Generate new airfoils.

        Args:
            num_samples: Number of airfoils to generate
            method: 'gmm' (best), 'empirical' (sample from training), or 'gaussian'
            diversity: Sampling temperature (1.0=normal, >1=more diverse, <1=safer)

        Returns:
            airfoils: (num_samples, 400) tensor
            latent_codes: (num_samples, latent_dim) tensor
        """

        if method == 'gmm':
            # Sample from fitted GMM (best quality + diversity)
            z = self.gmm.sample(num_samples)[0]
            z = z * diversity  # Scale for diversity control

        elif method == 'empirical':
            # Sample directly from training latent codes (safest)
            indices = np.random.randint(0, len(self.latent_codes), size=num_samples)
            z = self.latent_codes[indices]

            # Add small noise for diversity
            noise = np.random.randn(*z.shape) * 0.05 * diversity
            z = z + noise

        elif method == 'gaussian':
            # Sample from Gaussian fit to latent codes (simple)
            mean = self.latent_codes.mean(axis=0)
            std = self.latent_codes.std(axis=0)
            z = np.random.randn(num_samples, self.latent_dim) * std * diversity + mean

        else:
            raise ValueError(f"Unknown method: {method}")

        z_tensor = torch.tensor(z, dtype=torch.float32)

        with torch.no_grad():
            airfoils = self.model.decode(z_tensor)

        return airfoils, z_tensor

    def validate_airfoil(self, coords):
        """
        Validate airfoil quality.

        Returns:
            dict with quality metrics
        """
        # Trailing edge closure
        te_gap = np.linalg.norm(coords[0] - coords[-1])

        # Smoothness (second derivative)
        first_diff = np.diff(coords, axis=0)
        second_diff = np.diff(first_diff, axis=0)
        smoothness = np.mean(np.linalg.norm(second_diff, axis=1))

        # Check if points are mostly monotonic in x (no major loops)
        x_coords = coords[:, 0]
        # Allow small violations
        monotonic_top = np.sum(np.diff(x_coords[:100]) < -0.01) < 5
        monotonic_bot = np.sum(np.diff(x_coords[100:]) > 0.01) < 5

        return {
            'te_gap': te_gap,
            'smoothness': smoothness,
            'monotonic': monotonic_top and monotonic_bot,
            'valid': te_gap < 0.05 and smoothness < 0.01  # Relaxed validation
        }

    def generate_batch(self, num_samples=10, method='gmm', diversity=1.0,
                      max_attempts=100, return_invalid=False):
        """
        Generate batch of validated airfoils.

        Automatically retries invalid samples until we have enough valid ones.
        """
        valid_airfoils = []
        valid_z = []
        invalid_count = 0

        attempts = 0
        while len(valid_airfoils) < num_samples and attempts < max_attempts:
            # Generate candidates
            batch_size = min(20, num_samples * 2)
            airfoils, z = self.generate(batch_size, method=method, diversity=diversity)

            # Validate each
            for i in range(len(airfoils)):
                coords = airfoils[i].numpy().reshape(200, 2)
                metrics = self.validate_airfoil(coords)

                if metrics['valid']:
                    valid_airfoils.append(airfoils[i])
                    valid_z.append(z[i])
                else:
                    invalid_count += 1

                if len(valid_airfoils) >= num_samples:
                    break

            attempts += 1

        if len(valid_airfoils) < num_samples:
            print(f"Warning: Only generated {len(valid_airfoils)}/{num_samples} valid airfoils")

        valid_airfoils = torch.stack(valid_airfoils[:num_samples])
        valid_z = torch.stack(valid_z[:num_samples])

        print(f"Generated {len(valid_airfoils)} valid airfoils ({invalid_count} invalid rejected)")

        return valid_airfoils, valid_z


def demonstrate_generator():
    """Demonstrate production-ready generator."""

    print("=" * 70)
    print("Production Airfoil Generator Demo")
    print("=" * 70)

    # Initialize
    gen = AirfoilGenerator()

    # Create output directory
    output_dir = Path("outputs/generated_airfoils")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: GMM sampling (BEST)
    print("\n" + "=" * 70)
    print("Test 1: GMM Sampling (Recommended)")
    print("=" * 70)

    airfoils_gmm, z_gmm = gen.generate_batch(num_samples=10, method='gmm', diversity=1.0)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Generated Airfoils - GMM Sampling (Production Quality)', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i in range(10):
        coords = airfoils_gmm[i].numpy().reshape(200, 2)
        metrics = gen.validate_airfoil(coords)

        ax = axes[i]
        ax.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=1.5)
        ax.set_aspect('equal')
        ax.set_title(f'Generated {i+1}\nTE: {metrics["te_gap"]:.4f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.3, 0.3)
        ax.set_xlabel('x/c')
        if i % 5 == 0:
            ax.set_ylabel('y/c')

    plt.tight_layout()
    plt.savefig(output_dir / 'gmm_generated.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'gmm_generated.png'}")
    plt.close()

    # Test 2: Diversity control
    print("\n" + "=" * 70)
    print("Test 2: Diversity Control")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Diversity Control (diversity parameter)', fontsize=14, fontweight='bold')

    for idx, (diversity, title) in enumerate([(0.5, 'Conservative'), (1.0, 'Normal'), (1.5, 'Diverse')]):
        airfoils, _ = gen.generate(num_samples=5, method='gmm', diversity=diversity)

        ax = axes[idx]
        for i in range(5):
            coords = airfoils[i].numpy().reshape(200, 2)
            ax.plot(coords[:, 0], coords[:, 1], linewidth=1.5, alpha=0.7)

        ax.set_aspect('equal')
        ax.set_title(f'{title} (diversity={diversity})', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.3, 0.3)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')

    plt.tight_layout()
    plt.savefig(output_dir / 'diversity_control.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'diversity_control.png'}")
    plt.close()

    # Test 3: Method comparison
    print("\n" + "=" * 70)
    print("Test 3: Sampling Method Comparison")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Sampling Method Comparison', fontsize=14, fontweight='bold')

    for idx, (method, title) in enumerate([('gmm', 'GMM'), ('gaussian', 'Gaussian'), ('empirical', 'Empirical')]):
        airfoils, _ = gen.generate(num_samples=5, method=method, diversity=1.0)

        ax = axes[idx]
        for i in range(5):
            coords = airfoils[i].numpy().reshape(200, 2)
            ax.plot(coords[:, 0], coords[:, 1], linewidth=1.5, alpha=0.7)

        ax.set_aspect('equal')
        ax.set_title(f'{title} Method', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.3, 0.3)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')

    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'method_comparison.png'}")
    plt.close()

    print("\n" + "=" * 70)
    print("Demo complete! All outputs in: outputs/generated_airfoils/")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_generator()
