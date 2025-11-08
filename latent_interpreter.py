"""
latent_interpreter.py

Practical latent space interpretation system for the working AE.
Discovers what each dimension controls through empirical analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from scipy.interpolate import interp1d

from src.models.airfoil_ae import AirfoilAE
from src.data.create_dataset import AirfoilDataset


class LatentInterpreter:
    """
    Interpret latent dimensions by analyzing their effects on airfoil geometry.
    """

    def __init__(self, model_path="models/airfoil_ae/best_model.pth",
                 dataset_path="data/processed/airfoil_dataset.pkl"):
        """Initialize interpreter with trained model."""

        print("Initializing Latent Interpreter...")

        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.latent_dim = checkpoint.get('latent_dim', 24)

        self.model = AirfoilAE(latent_dim=self.latent_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load dataset
        self.dataset = AirfoilDataset.load(dataset_path)

        print(f"  Model loaded (latent_dim={self.latent_dim})")
        print(f"  Dataset loaded ({len(self.dataset)} airfoils)")

        # Encode all airfoils and extract features
        self._analyze_latent_space()

    def _analyze_latent_space(self):
        """Encode all airfoils and extract geometric features."""

        print("\nAnalyzing latent space...")

        self.latent_codes = []
        self.features = []

        with torch.no_grad():
            for i in range(len(self.dataset)):
                airfoil = self.dataset[i].unsqueeze(0)
                z = self.model.encode(airfoil)
                self.latent_codes.append(z)

                # Extract geometric features
                coords = airfoil.numpy().reshape(200, 2)
                features = self._extract_features(coords)
                self.features.append(features)

                if (i + 1) % 500 == 0:
                    print(f"  Analyzed {i + 1}/{len(self.dataset)} airfoils...")

        self.latent_codes = torch.cat(self.latent_codes, dim=0).numpy()  # (N, latent_dim)
        self.features = np.array(self.features)  # (N, num_features)

        # Compute correlations between latent dims and features
        self._compute_dimension_effects()

        print("  Analysis complete!")

    def _extract_features(self, coords):
        """Extract simple geometric features from airfoil."""

        # Separate upper and lower surfaces
        upper = coords[:100]
        lower = coords[100:]

        # Thickness metrics
        max_thickness = np.max(upper[:, 1] - lower[:, 1])
        avg_thickness = np.mean(upper[:, 1] - lower[:, 1])

        # Thickness location
        thickness_dist = upper[:, 1] - lower[:, 1]
        max_thick_loc = upper[np.argmax(thickness_dist), 0]

        # Camber metrics
        camber_line = (upper[:, 1] + lower[:, 1]) / 2
        max_camber = np.max(np.abs(camber_line))
        avg_camber = np.mean(camber_line)

        # Leading edge radius (approximate)
        le_curve = np.max(coords[:10, 1]) - np.min(coords[:10, 1])

        # Trailing edge
        te_thickness = np.abs(coords[0, 1] - coords[-1, 1])

        # Overall shape
        aspect_ratio = np.ptp(coords[:, 0]) / np.ptp(coords[:, 1])

        return np.array([
            max_thickness,
            avg_thickness,
            max_thick_loc,
            max_camber,
            avg_camber,
            le_curve,
            te_thickness,
            aspect_ratio
        ])

    def _compute_dimension_effects(self):
        """Compute which latent dimensions control which features."""

        feature_names = [
            'max_thickness',
            'avg_thickness',
            'max_thick_location',
            'max_camber',
            'avg_camber',
            'le_curvature',
            'te_thickness',
            'aspect_ratio'
        ]

        print("\nComputing dimension-feature correlations...")

        self.correlations = np.zeros((self.latent_dim, len(feature_names)))

        for dim in range(self.latent_dim):
            for feat_idx in range(len(feature_names)):
                corr, _ = pearsonr(self.latent_codes[:, dim], self.features[:, feat_idx])
                self.correlations[dim, feat_idx] = corr

        # Find strongest effect for each dimension
        self.dimension_labels = []
        for dim in range(self.latent_dim):
            strongest_feat_idx = np.argmax(np.abs(self.correlations[dim]))
            strongest_corr = self.correlations[dim, strongest_feat_idx]
            self.dimension_labels.append({
                'dim': dim,
                'controls': feature_names[strongest_feat_idx],
                'correlation': strongest_corr,
                'strength': np.abs(strongest_corr)
            })

        # Sort by strength
        self.dimension_labels.sort(key=lambda x: x['strength'], reverse=True)

        print("\nTop 10 Most Interpretable Dimensions:")
        print("-" * 60)
        for i in range(min(10, len(self.dimension_labels))):
            info = self.dimension_labels[i]
            print(f"  Dim {info['dim']:2d}: {info['controls']:20s} (r={info['correlation']:+.3f})")

    def manipulate_dimension(self, base_airfoil_idx, dimension, strength, num_steps=7):
        """
        Manipulate a specific dimension and show the effect.

        Args:
            base_airfoil_idx: Index of base airfoil from dataset
            dimension: Which dimension to vary
            strength: How much to vary (+/- std devs)
            num_steps: Number of steps to show
        """

        base_airfoil = self.dataset[base_airfoil_idx].unsqueeze(0)

        with torch.no_grad():
            base_z = self.model.encode(base_airfoil)

        # Get dimension stats for meaningful variation
        dim_std = np.std(self.latent_codes[:, dimension])

        # Create variations
        values = np.linspace(-strength * dim_std, strength * dim_std, num_steps)

        fig, axes = plt.subplots(1, num_steps, figsize=(20, 3))
        fig.suptitle(f'Manipulating Dimension {dimension} '
                     f'({self.dimension_labels[dimension]["controls"]})',
                     fontsize=14, fontweight='bold')

        with torch.no_grad():
            for i, val in enumerate(values):
                z = base_z.clone()
                z[0, dimension] = base_z[0, dimension] + val

                airfoil = self.model.decode(z)
                coords = airfoil.numpy().reshape(200, 2)

                ax = axes[i]
                ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=1.5)
                ax.set_aspect('equal')
                ax.set_title(f'{val/dim_std:+.1f} std', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-0.1, 1.1)
                ax.set_ylim(-0.3, 0.3)
                ax.set_xlabel('x/c', fontsize=8)
                if i == 0:
                    ax.set_ylabel('y/c', fontsize=8)

        plt.tight_layout()
        return fig

    def generate_with_target_features(self, target_thickness=None, target_camber=None,
                                     num_samples=5):
        """
        Generate airfoils with specific target features.

        This finds existing airfoils with similar features and samples nearby.
        """

        # Find airfoils matching criteria
        mask = np.ones(len(self.features), dtype=bool)

        if target_thickness is not None:
            thickness_feat = self.features[:, 0]  # max_thickness
            thickness_err = np.abs(thickness_feat - target_thickness)
            mask &= thickness_err < 0.05

        if target_camber is not None:
            camber_feat = self.features[:, 3]  # max_camber
            camber_err = np.abs(camber_feat - target_camber)
            mask &= camber_err < 0.02

        matching_indices = np.where(mask)[0]

        if len(matching_indices) == 0:
            print(f"No airfoils found matching criteria")
            print(f"  target_thickness: {target_thickness}")
            print(f"  target_camber: {target_camber}")
            return None, None

        print(f"Found {len(matching_indices)} matching airfoils")

        # Sample from matching region with small noise
        selected = np.random.choice(matching_indices, size=min(num_samples, len(matching_indices)))
        base_latents = self.latent_codes[selected]

        # Add small noise for diversity
        noise = np.random.randn(*base_latents.shape) * 0.05
        new_latents = base_latents + noise

        # Decode
        with torch.no_grad():
            z_tensor = torch.tensor(new_latents, dtype=torch.float32)
            airfoils = self.model.decode(z_tensor)

        return airfoils, z_tensor

    def find_similar(self, reference_idx, num_results=5, feature_weights=None):
        """
        Find airfoils similar to reference in latent space.
        """

        if feature_weights is None:
            # Default: weight by feature importance
            feature_weights = np.ones(self.features.shape[1])

        ref_features = self.features[reference_idx]
        ref_latent = self.latent_codes[reference_idx]

        # Compute distances in latent space
        latent_dist = np.linalg.norm(self.latent_codes - ref_latent, axis=1)

        # Get top matches
        similar_indices = np.argsort(latent_dist)[1:num_results+1]  # Exclude self

        return similar_indices

    def visualize_dimension_effects(self, output_dir="outputs/latent_interpretation"):
        """Create comprehensive visualization of dimension effects."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(self.correlations.T, aspect='auto', cmap='RdBu_r',
                      vmin=-1, vmax=1)

        feature_names = ['max_thick', 'avg_thick', 'thick_loc', 'max_camber',
                        'avg_camber', 'le_curve', 'te_thick', 'aspect']

        ax.set_xticks(range(self.latent_dim))
        ax.set_yticks(range(len(feature_names)))
        ax.set_xticklabels([f'D{i}' for i in range(self.latent_dim)], fontsize=8)
        ax.set_yticklabels(feature_names, fontsize=10)

        ax.set_xlabel('Latent Dimension', fontsize=12)
        ax.set_ylabel('Geometric Feature', fontsize=12)
        ax.set_title('Latent Dimension → Feature Correlations', fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Pearson Correlation')
        plt.tight_layout()
        plt.savefig(output_dir / 'dimension_correlations.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nSaved: {output_dir / 'dimension_correlations.png'}")

        # 2. Top dimensions visualization
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        fig.suptitle('Top 10 Most Interpretable Dimensions', fontsize=14, fontweight='bold')

        for i in range(min(10, len(self.dimension_labels))):
            info = self.dimension_labels[i]
            dim = info['dim']

            # Show effect of this dimension
            base_idx = len(self.dataset) // 2  # Use middle airfoil
            self.manipulate_dimension(base_idx, dim, strength=2, num_steps=5)

            plt.savefig(output_dir / f'dim_{dim:02d}_effect.png', dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Saved top 10 dimension effects to: {output_dir}/")

        return output_dir


def demonstrate_interpreter():
    """Demonstrate latent interpretation system."""

    print("=" * 70)
    print("Latent Space Interpretation System")
    print("=" * 70)

    # Initialize
    interp = LatentInterpreter()

    # Visualize dimension effects
    output_dir = interp.visualize_dimension_effects()

    # Test targeted generation
    print("\n" + "=" * 70)
    print("Test: Generate Thick Airfoils")
    print("=" * 70)

    airfoils, z = interp.generate_with_target_features(target_thickness=0.15, num_samples=5)

    if airfoils is not None:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle('Generated Thick Airfoils (thickness ≈ 0.15)', fontsize=14, fontweight='bold')

        for i in range(len(airfoils)):
            coords = airfoils[i].numpy().reshape(200, 2)
            ax = axes[i]
            ax.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=2)
            ax.set_aspect('equal')
            ax.set_title(f'Sample {i+1}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.3, 0.3)
            ax.set_xlabel('x/c')
            if i == 0:
                ax.set_ylabel('y/c')

        plt.tight_layout()
        plt.savefig(output_dir / 'targeted_thick_airfoils.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_dir / 'targeted_thick_airfoils.png'}")

    print("\n" + "=" * 70)
    print("Interpretation complete!")
    print(f"All outputs saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_interpreter()
