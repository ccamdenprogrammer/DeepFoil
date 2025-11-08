"""
Custom Workflow Example

This script demonstrates advanced usage of DeepFoil:
- Using the AirfoilGenerator for diverse sampling
- Filtering results based on custom criteria
- Custom visualization
- Integration with your own analysis pipeline

Run from the DEEPFOIL_DISTRIBUTION directory:
    python examples/custom_workflow.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from latent_interpreter import LatentInterpreter
from generate_airfoils import AirfoilGenerator
import numpy as np
import matplotlib.pyplot as plt


def calculate_metrics(coords):
    """Calculate geometric metrics for an airfoil."""
    # Split into upper and lower surfaces
    upper = coords[:100]
    lower = coords[100:]

    # Thickness
    thickness_dist = upper[:, 1] - lower[:, 1]
    max_thickness = np.max(thickness_dist)
    max_thickness_loc = upper[np.argmax(thickness_dist), 0]

    # Camber
    camber_line = (upper[:, 1] + lower[:, 1]) / 2
    max_camber = np.max(np.abs(camber_line))
    max_camber_loc = upper[np.argmax(np.abs(camber_line)), 0]

    # Leading edge radius (approximate)
    le_idx = np.argmin(coords[:, 0])
    le_curvature = estimate_curvature(coords, le_idx)
    le_radius = 1.0 / (le_curvature + 1e-10)

    # Trailing edge gap
    te_gap = np.abs(coords[0, 1] - coords[-1, 1])

    return {
        'max_thickness': max_thickness,
        'max_thickness_loc': max_thickness_loc,
        'max_camber': max_camber,
        'max_camber_loc': max_camber_loc,
        'le_radius': le_radius,
        'te_gap': te_gap
    }


def estimate_curvature(coords, idx, window=5):
    """Estimate curvature at a point using nearby points."""
    start = max(0, idx - window)
    end = min(len(coords), idx + window + 1)
    local_coords = coords[start:end]

    if len(local_coords) < 3:
        return 0.0

    # Fit circle through points (simplified)
    x = local_coords[:, 0]
    y = local_coords[:, 1]

    # Use finite differences for curvature approximation
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

    return np.mean(curvature)


def custom_filter(coords, criteria):
    """Filter airfoil based on custom criteria."""
    metrics = calculate_metrics(coords)

    # Check each criterion
    for key, (min_val, max_val) in criteria.items():
        if key not in metrics:
            continue
        if metrics[key] < min_val or metrics[key] > max_val:
            return False, metrics

    return True, metrics


def create_detailed_visualization(selected_airfoils, output_dir):
    """Create detailed visualization with metrics."""
    num_airfoils = len(selected_airfoils)

    fig = plt.figure(figsize=(20, num_airfoils * 6))
    gs = fig.add_gridspec(num_airfoils, 3, width_ratios=[2, 1, 1])

    for i, airfoil_data in enumerate(selected_airfoils):
        coords = airfoil_data['coords']
        metrics = airfoil_data['metrics']

        # Main airfoil plot
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2.5)
        ax1.fill_between(coords[:, 0], coords[:, 1], alpha=0.2)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x/c', fontsize=11)
        ax1.set_ylabel('y/c', fontsize=11)
        ax1.set_title(f'Airfoil {i+1}', fontsize=13, fontweight='bold')
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.15, 0.15)

        # Mark max thickness and camber locations
        ax1.axvline(metrics['max_thickness_loc'], color='r', linestyle='--',
                   alpha=0.5, label=f"Max t/c location")
        ax1.axvline(metrics['max_camber_loc'], color='g', linestyle='--',
                   alpha=0.5, label=f"Max camber location")
        ax1.legend(fontsize=9)

        # Thickness distribution
        ax2 = fig.add_subplot(gs[i, 1])
        upper = coords[:100]
        lower = coords[100:]
        thickness_dist = upper[:, 1] - lower[:, 1]
        ax2.plot(upper[:, 0], thickness_dist, 'r-', linewidth=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('x/c', fontsize=10)
        ax2.set_ylabel('Thickness', fontsize=10)
        ax2.set_title('Thickness Distribution', fontsize=11)
        ax2.axhline(metrics['max_thickness'], color='r', linestyle='--', alpha=0.5)

        # Metrics table
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.axis('off')

        metrics_text = [
            f"GEOMETRIC METRICS",
            f"",
            f"Max Thickness:  {metrics['max_thickness']:.2%}",
            f"  Location:     {metrics['max_thickness_loc']:.2f}",
            f"",
            f"Max Camber:     {metrics['max_camber']:.2%}",
            f"  Location:     {metrics['max_camber_loc']:.2f}",
            f"",
            f"LE Radius:      {metrics['le_radius']:.4f}",
            f"TE Gap:         {metrics['te_gap']:.4f}",
        ]

        ax3.text(0.1, 0.9, '\n'.join(metrics_text), fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_file = output_dir / 'custom_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved detailed visualization: {output_file}")


def main():
    print("=" * 80)
    print("CUSTOM WORKFLOW EXAMPLE")
    print("=" * 80)

    # Initialize both interpreter and generator
    print("\n1. Initializing DeepFoil system...")
    interpreter = LatentInterpreter()
    generator = AirfoilGenerator()
    print("   System ready!")

    # Step 1: Generate diverse candidates
    print("\n2. Generating diverse airfoil candidates...")
    print("   Method: GMM sampling (best quality)")

    num_candidates = 50
    airfoils, latent_codes = generator.generate_batch(
        num_samples=num_candidates,
        method='gmm',
        diversity=1.2  # Higher diversity
    )

    print(f"   Generated {len(airfoils)} candidates")

    # Step 2: Apply custom filtering criteria
    print("\n3. Applying custom filtering criteria...")

    criteria = {
        'max_thickness': (0.10, 0.15),      # 10-15% thickness
        'max_thickness_loc': (0.25, 0.40),  # Thickness peak at 25-40% chord
        'max_camber': (0.02, 0.04),         # 2-4% camber
        'te_gap': (0.0, 0.002),             # Small trailing edge gap
    }

    print("   Criteria:")
    for key, (min_val, max_val) in criteria.items():
        print(f"     {key}: {min_val:.4f} - {max_val:.4f}")

    selected_airfoils = []

    for i, airfoil in enumerate(airfoils):
        coords = airfoil.numpy().reshape(200, 2)
        passes, metrics = custom_filter(coords, criteria)

        if passes:
            selected_airfoils.append({
                'index': i,
                'coords': coords,
                'metrics': metrics,
                'latent_code': latent_codes[i]
            })

    print(f"\n   {len(selected_airfoils)} out of {len(airfoils)} airfoils passed filtering")
    print(f"   Success rate: {100*len(selected_airfoils)/len(airfoils):.1f}%")

    if len(selected_airfoils) == 0:
        print("\n   No airfoils met criteria. Try relaxing the constraints.")
        return

    # Step 3: Rank by custom scoring
    print("\n4. Ranking airfoils by custom score...")

    for airfoil_data in selected_airfoils:
        metrics = airfoil_data['metrics']

        # Custom scoring function (example: prefer thickness at 30% chord)
        thickness_loc_score = 1.0 - abs(metrics['max_thickness_loc'] - 0.30) / 0.30
        camber_score = metrics['max_camber'] / 0.04  # Normalize to max
        te_closure_score = 1.0 - metrics['te_gap'] / 0.002

        total_score = (thickness_loc_score * 0.4 +
                      camber_score * 0.3 +
                      te_closure_score * 0.3)

        airfoil_data['score'] = total_score

    # Sort by score
    selected_airfoils.sort(key=lambda x: x['score'], reverse=True)

    print("   Top 5 airfoils:")
    for i in range(min(5, len(selected_airfoils))):
        airfoil_data = selected_airfoils[i]
        print(f"     {i+1}. Score: {airfoil_data['score']:.3f} - " +
              f"t/c={airfoil_data['metrics']['max_thickness']:.2%} @ " +
              f"{airfoil_data['metrics']['max_thickness_loc']:.2f}")

    # Step 4: Save results
    print("\n5. Saving results...")

    output_dir = Path("examples_output/custom_workflow")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save top N airfoils
    top_n = min(10, len(selected_airfoils))

    for i in range(top_n):
        airfoil_data = selected_airfoils[i]
        coords = airfoil_data['coords']

        # Save .dat file
        dat_file = output_dir / f"ranked_{i+1:02d}.dat"
        with open(dat_file, 'w') as f:
            f.write(f"DeepFoil Custom Workflow - Rank {i+1}\n")
            f.write(f"Score: {airfoil_data['score']:.3f}\n")
            for x, y in coords:
                f.write(f"{x:.6f}  {y:.6f}\n")

    print(f"   Saved top {top_n} airfoils to: {output_dir}/")

    # Create detailed visualization for top 5
    print("\n6. Creating detailed visualization...")
    top_5 = selected_airfoils[:min(5, len(selected_airfoils))]
    create_detailed_visualization(top_5, output_dir)

    # Save report
    print("\n7. Creating analysis report...")
    report_file = output_dir / 'CUSTOM_WORKFLOW_REPORT.txt'

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DEEPFOIL CUSTOM WORKFLOW REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("FILTERING CRITERIA:\n")
        f.write("-" * 80 + "\n")
        for key, (min_val, max_val) in criteria.items():
            f.write(f"{key:20s}: {min_val:.4f} - {max_val:.4f}\n")

        f.write(f"\nRESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Candidates generated: {len(airfoils)}\n")
        f.write(f"Airfoils passed filter: {len(selected_airfoils)}\n")
        f.write(f"Success rate: {100*len(selected_airfoils)/len(airfoils):.1f}%\n")

        f.write(f"\nTOP {top_n} AIRFOILS:\n")
        f.write("-" * 80 + "\n")

        for i in range(top_n):
            airfoil_data = selected_airfoils[i]
            m = airfoil_data['metrics']

            f.write(f"\nRank {i+1} (Score: {airfoil_data['score']:.3f}):\n")
            f.write(f"  File: ranked_{i+1:02d}.dat\n")
            f.write(f"  Max Thickness: {m['max_thickness']:.2%} at {m['max_thickness_loc']:.2f}\n")
            f.write(f"  Max Camber: {m['max_camber']:.2%} at {m['max_camber_loc']:.2f}\n")
            f.write(f"  LE Radius: {m['le_radius']:.4f}\n")
            f.write(f"  TE Gap: {m['te_gap']:.4f}\n")

    print(f"   Saved report: {report_file}")

    # Final summary
    print("\n" + "=" * 80)
    print("CUSTOM WORKFLOW COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated {num_candidates} candidates")
    print(f"Selected {len(selected_airfoils)} based on criteria")
    print(f"Saved top {top_n} ranked airfoils")
    print(f"\nOutput directory: {output_dir}/")
    print("\nThis workflow demonstrated:")
    print("  - Diverse candidate generation")
    print("  - Custom filtering criteria")
    print("  - Custom scoring/ranking")
    print("  - Detailed geometric analysis")
    print("  - Advanced visualization")
    print("\nModify this script to implement your own design criteria!")
    print()


if __name__ == "__main__":
    main()
