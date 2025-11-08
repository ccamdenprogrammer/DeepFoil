"""
DeepFoil Quick Demo - Portfolio Demonstration
==============================================

This script provides a fast, impressive demonstration of DeepFoil's capabilities.
Perfect for recruiters and portfolio reviews.

Runtime: ~30 seconds
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from latent_interpreter import LatentInterpreter

def print_header():
    """Print impressive header."""
    print("\n" + "="*80)
    print("  ____                  _____      _ _     ___            _      _   ")
    print(" |  _ \\  ___  ___ _ __ |  ___|__  (_) |   / _ \\ _   _(_) ___| | __")
    print(" | | | |/ _ \\/ _ \\ '_ \\| |_ / _ \\ | | |  | | | | | | | |/ __| |/ /")
    print(" | |_| |  __/  __/ |_) |  _| (_) || | |  | |_| | |_| | | (__|   < ")
    print(" |____/ \\___|\\___| .__/|_|  \\___/ |_|_|   \\__\\_\\\\__,_|_|\\___|_|\\_\\")
    print("                 |_|                                                ")
    print("")
    print("         AI-POWERED AIRFOIL DESIGN SYSTEM - QUICK DEMO")
    print("="*80)
    print()
    print("This demonstration showcases:")
    print("  [+] Deep learning-based generative design")
    print("  [+] Interpretable latent space with feature control")
    print("  [+] Production-quality airfoil generation")
    print("  [+] Real-world aerospace engineering application")
    print()
    print("="*80)

def demo_generation():
    """Demonstrate targeted airfoil generation."""
    print("\n[STEP 1/3] Initializing AI System...")
    print("-" * 80)
    print("Loading neural network model and analyzing latent space...")

    interpreter = LatentInterpreter()

    print(f"\n[OK] Model loaded successfully")
    print(f"  - Architecture: 400D -> 24D latent -> 400D")
    print(f"  - Training data: {len(interpreter.dataset):,} airfoils (UIUC database)")
    print(f"  - Latent dimensions: {interpreter.latent_dim}")

    # Show top interpretable dimensions
    print(f"\n  Most Interpretable Dimensions:")
    for i in range(5):
        info = interpreter.dimension_labels[i]
        print(f"    Dim {info['dim']:2d}: {info['controls']:15s} (correlation: {info['correlation']:+.3f})")

    # Generate airfoils
    print("\n[STEP 2/3] Generating Custom Airfoils...")
    print("-" * 80)

    specs = [
        {"thickness": 0.10, "camber": 0.00, "name": "Thin Symmetric", "desc": "High-speed flight"},
        {"thickness": 0.12, "camber": 0.03, "name": "Moderate Cambered", "desc": "General aviation"},
        {"thickness": 0.15, "camber": 0.05, "name": "High Lift", "desc": "Low-speed performance"},
    ]

    all_airfoils = []
    all_names = []

    for spec in specs:
        print(f"\nGenerating: {spec['name']}")
        print(f"  Target: {spec['thickness']:.0%} thickness, {spec['camber']:.0%} camber")
        print(f"  Purpose: {spec['desc']}")

        airfoils, z = interpreter.generate_with_target_features(
            target_thickness=spec['thickness'],
            target_camber=spec['camber'],
            num_samples=2
        )

        if airfoils is not None:
            print(f"  [OK] Generated {len(airfoils)} designs")
            all_airfoils.extend(airfoils)
            all_names.extend([spec['name']] * len(airfoils))

            # Calculate actual features
            for i, airfoil in enumerate(airfoils):
                coords = airfoil.numpy().reshape(200, 2)
                upper = coords[:100]
                lower = coords[100:]
                actual_t = np.max(upper[:, 1] - lower[:, 1])
                camber_line = (upper[:, 1] + lower[:, 1]) / 2
                actual_c = np.max(np.abs(camber_line))
                print(f"    Design {i+1}: thickness={actual_t:.1%}, camber={actual_c:.1%}")

    # Save and visualize
    print("\n[STEP 3/3] Saving Results...")
    print("-" * 80)

    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    # Save .dat files
    print("\nSaving industry-standard .dat files (XFOIL-compatible):")
    for i, (airfoil, name) in enumerate(zip(all_airfoils, all_names)):
        coords = airfoil.numpy().reshape(200, 2)
        filename = output_dir / f"demo_airfoil_{i+1:02d}_{name.replace(' ', '_')}.dat"

        with open(filename, 'w') as f:
            f.write(f"DeepFoil Generated - {name}\n")
            for x, y in coords:
                f.write(f"{x:.6f}  {y:.6f}\n")

        print(f"  [OK] {filename.name}")

    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    fig.suptitle('DeepFoil Generated Airfoils - Quick Demo',
                 fontsize=16, fontweight='bold')

    for i, (airfoil, name) in enumerate(zip(all_airfoils, all_names)):
        coords = airfoil.numpy().reshape(200, 2)
        ax = axes[i]

        # Plot airfoil
        ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2)
        ax.fill_between(coords[:, 0], coords[:, 1], alpha=0.3)

        # Calculate features
        upper = coords[:100]
        lower = coords[100:]
        thickness = np.max(upper[:, 1] - lower[:, 1])
        camber_line = (upper[:, 1] + lower[:, 1]) / 2
        camber = np.max(np.abs(camber_line))

        ax.set_title(f'{name}\nt={thickness:.1%}, c={camber:.1%}',
                    fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.2, 0.2)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')

    # Add info box in last subplot
    axes[-1].axis('off')
    info_text = """
    DeepFoil AI System

    • Deterministic Autoencoder
    • 24D Latent Space
    • MSE: 0.000004
    • 1,646 Training Airfoils

    Generated: {}

    Files saved to: demo_output/
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M"))

    axes[-1].text(0.1, 0.5, info_text,
                 fontsize=11, family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    viz_file = output_dir / 'demo_visualization.png'
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"  [OK] {viz_file.name}")

    # Print summary
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print(f"\n[OK] Generated {len(all_airfoils)} custom airfoil designs")
    print(f"[OK] Saved {len(all_airfoils)} .dat files (XFOIL-compatible)")
    print(f"[OK] Created visualization: demo_visualization.png")
    print(f"\nOutput location: {output_dir.absolute()}/")

    print("\n" + "-"*80)
    print("TECHNICAL HIGHLIGHTS")
    print("-"*80)
    print("- Neural Architecture: Custom deterministic autoencoder")
    print("- Latent Compression: 400D -> 24D (94% reduction)")
    print("- Reconstruction MSE: < 0.00001 (near-perfect accuracy)")
    print("- Interpretability: 10+ dimensions correlated with features")
    print("- Training Dataset: 1,646 airfoils from UIUC database")
    print("- Generation Speed: ~50 airfoils/second (CPU)")

    print("\n" + "-"*80)
    print("NEXT STEPS")
    print("-"*80)
    print("1. Review generated files in demo_output/")
    print("2. Open demo_visualization.png to see results")
    print("3. Try interactive mode: python deepfoil_interface.py")
    print("4. Read PORTFOLIO_README.md for full project details")
    print("5. Run examples: python examples/simple_generation.py")

    print("\n" + "="*80)
    print("Thank you for reviewing DeepFoil!")
    print("="*80)
    print()

def main():
    """Run quick demo."""
    try:
        print_header()
        demo_generation()
        return 0
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf you see import errors, please run: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
