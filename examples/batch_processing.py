"""
Batch Processing Example

This script demonstrates how to generate multiple sets of airfoils
with different specifications in a single run.

Run from the DEEPFOIL_DISTRIBUTION directory:
    python examples/batch_processing.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from latent_interpreter import LatentInterpreter
import numpy as np
import matplotlib.pyplot as plt


def save_airfoil_batch(airfoils, spec_name, output_dir, spec_info):
    """Save a batch of airfoils to a subdirectory."""
    batch_dir = output_dir / spec_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Saving {len(airfoils)} airfoils to: {batch_dir}/")

    results = []

    for i, airfoil in enumerate(airfoils):
        coords = airfoil.numpy().reshape(200, 2)

        # Save .dat file
        dat_file = batch_dir / f"{spec_name}_{i+1:02d}.dat"
        with open(dat_file, 'w') as f:
            f.write(f"DeepFoil - {spec_name}\n")
            f.write(f"Specification: {spec_info}\n")
            for x, y in coords:
                f.write(f"{x:.6f}  {y:.6f}\n")

        # Calculate metrics
        upper = coords[:100]
        lower = coords[100:]
        thickness = np.max(upper[:, 1] - lower[:, 1])
        camber_line = (upper[:, 1] + lower[:, 1]) / 2
        camber = np.max(np.abs(camber_line))

        results.append({
            'index': i+1,
            'thickness': thickness,
            'camber': camber,
            'coords': coords
        })

    return results, batch_dir


def create_visualization(all_results, output_dir):
    """Create a comparison visualization of all generated batches."""
    print("\n  Creating visualization...")

    num_batches = len(all_results)
    fig, axes = plt.subplots(num_batches, 5, figsize=(20, num_batches*4))

    if num_batches == 1:
        axes = axes.reshape(1, -1)

    for batch_idx, (spec_name, results) in enumerate(all_results.items()):
        for i in range(min(5, len(results))):
            ax = axes[batch_idx, i]
            coords = results[i]['coords']

            ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2)
            ax.fill_between(coords[:, 0], coords[:, 1], alpha=0.3)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.15, 0.15)

            if i == 0:
                ax.set_ylabel(spec_name, fontsize=10, fontweight='bold')

            title = f"t/c={results[i]['thickness']:.1%}\nc={results[i]['camber']:.1%}"
            ax.set_title(title, fontsize=9)

        # Hide extra subplots if less than 5 airfoils
        for i in range(len(results), 5):
            axes[batch_idx, i].axis('off')

    plt.tight_layout()
    output_file = output_dir / 'batch_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualization: {output_file}")


def main():
    print("=" * 80)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 80)

    # Initialize
    print("\n1. Loading model...")
    interpreter = LatentInterpreter()
    print(f"   Model loaded! ({len(interpreter.dataset)} training airfoils)")

    # Define multiple specifications
    specifications = [
        {
            'name': 'thin_symmetric',
            'thickness': 0.08,
            'camber': 0.00,
            'num_samples': 5,
            'description': 'Thin symmetric (high-speed)'
        },
        {
            'name': 'moderate_lift',
            'thickness': 0.12,
            'camber': 0.03,
            'num_samples': 5,
            'description': 'Moderate thickness and camber (general purpose)'
        },
        {
            'name': 'high_lift',
            'thickness': 0.15,
            'camber': 0.05,
            'num_samples': 5,
            'description': 'Thick with high camber (low-speed, high-lift)'
        },
        {
            'name': 'thick_symmetric',
            'thickness': 0.18,
            'camber': 0.00,
            'num_samples': 5,
            'description': 'Very thick symmetric (structural)'
        },
    ]

    print(f"\n2. Processing {len(specifications)} different specifications...")

    # Output directory
    output_dir = Path("examples_output/batch_processing")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each specification
    all_results = {}

    for spec in specifications:
        print(f"\n   Processing: {spec['name']}")
        print(f"   Description: {spec['description']}")
        print(f"   Target: thickness={spec['thickness']:.1%}, camber={spec['camber']:.1%}")

        # Generate
        airfoils, z = interpreter.generate_with_target_features(
            target_thickness=spec['thickness'],
            target_camber=spec['camber'],
            num_samples=spec['num_samples']
        )

        if airfoils is None:
            print(f"   WARNING: No matching airfoils found for {spec['name']}")
            continue

        # Save
        spec_info = f"thickness={spec['thickness']:.1%}, camber={spec['camber']:.1%}"
        results, batch_dir = save_airfoil_batch(
            airfoils,
            spec['name'],
            output_dir,
            spec_info
        )

        all_results[spec['name']] = results

        # Print summary statistics
        thicknesses = [r['thickness'] for r in results]
        cambers = [r['camber'] for r in results]

        print(f"   Generated {len(results)} airfoils:")
        print(f"     Thickness: {np.mean(thicknesses):.2%} ± {np.std(thicknesses):.2%}")
        print(f"     Camber: {np.mean(cambers):.2%} ± {np.std(cambers):.2%}")

    # Create comparison visualization
    print("\n3. Creating comparison visualization...")
    create_visualization(all_results, output_dir)

    # Create summary report
    print("\n4. Creating summary report...")
    report_file = output_dir / 'BATCH_REPORT.txt'

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DEEPFOIL BATCH PROCESSING REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("SPECIFICATIONS PROCESSED:\n")
        f.write("-" * 80 + "\n\n")

        for spec in specifications:
            if spec['name'] in all_results:
                f.write(f"{spec['name']}:\n")
                f.write(f"  Description: {spec['description']}\n")
                f.write(f"  Target: thickness={spec['thickness']:.1%}, camber={spec['camber']:.1%}\n")
                f.write(f"  Generated: {len(all_results[spec['name']])} airfoils\n\n")

        f.write("\nDETAILED RESULTS:\n")
        f.write("-" * 80 + "\n\n")

        for spec_name, results in all_results.items():
            f.write(f"{spec_name}:\n")
            for r in results:
                f.write(f"  Airfoil {r['index']:02d}: ")
                f.write(f"thickness={r['thickness']:.2%}, ")
                f.write(f"camber={r['camber']:.2%}\n")
            f.write("\n")

    print(f"   Saved report: {report_file}")

    # Final summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nTotal specifications processed: {len(all_results)}")
    print(f"Total airfoils generated: {sum(len(r) for r in all_results.values())}")
    print(f"\nOutput directory: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - {len(all_results)} subdirectories (one per specification)")
    print(f"  - .dat files for each airfoil")
    print(f"  - batch_comparison.png (visual comparison)")
    print(f"  - BATCH_REPORT.txt (summary)")

    print("\nNext steps:")
    print("  1. Review batch_comparison.png to see all designs")
    print("  2. Select best candidates from each category")
    print("  3. Run XFOIL analysis on selected airfoils")
    print("  4. Modify specifications in this script to explore more designs")
    print()


if __name__ == "__main__":
    main()
