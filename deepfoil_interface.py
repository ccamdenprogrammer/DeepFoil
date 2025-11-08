"""
DeepFoil - Interpretable Latent Space Airfoil Generation System
Interactive Command-Line Interface

This is the main entry point for the DeepFoil system.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from latent_interpreter import LatentInterpreter
from generate_airfoils import AirfoilGenerator


class DeepFoilInterface:
    """Interactive interface for DeepFoil system."""

    def __init__(self):
        self.interpreter = None
        self.generator = None
        self.output_dir = Path("deepfoil_outputs")
        self.output_dir.mkdir(exist_ok=True)

    def clear_screen(self):
        """Clear the console screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """Print the application header."""
        print("=" * 80)
        print("  ____                  _____      _ _ ")
        print(" |  _ \\  ___  ___ _ __ |  ___|__  (_) |")
        print(" | | | |/ _ \\/ _ \\ '_ \\| |_ / _ \\ | | |")
        print(" | |_| |  __/  __/ |_) |  _| (_) || | |")
        print(" |____/ \\___|\\___| .__/|_|  \\___/ |_|_|")
        print("                 |_|                    ")
        print()
        print("  Interpretable Latent Space Airfoil Generation System")
        print("  Version 1.0")
        print("=" * 80)

    def print_menu(self):
        """Print the main menu."""
        print("\n" + "=" * 80)
        print("MAIN MENU")
        print("=" * 80)
        print("1. Initialize System (Load Model & Dataset)")
        print("2. Generate Airfoils with Target Features")
        print("3. Explore Dimension Effects (Thickness/Camber)")
        print("4. Analyze Latent Space (View Correlations)")
        print("5. Batch Generate Airfoils")
        print("6. Design Workflow (Guided)")
        print("7. Settings")
        print("8. Help & Documentation")
        print("9. Exit")
        print("=" * 80)

    def initialize_system(self):
        """Initialize the latent interpreter and generator."""
        print("\n" + "=" * 80)
        print("INITIALIZING DEEPFOIL SYSTEM")
        print("=" * 80)

        try:
            print("\nLoading latent interpreter...")
            self.interpreter = LatentInterpreter()

            print("\nLoading airfoil generator...")
            self.generator = AirfoilGenerator()

            print("\n" + "=" * 80)
            print("SUCCESS! System ready.")
            print("=" * 80)
            print(f"\nLatent dimensions: {self.interpreter.latent_dim}")
            print(f"Training airfoils: {len(self.interpreter.dataset)}")
            print("\nTop interpretable dimensions:")
            for i in range(min(5, len(self.interpreter.dimension_labels))):
                info = self.interpreter.dimension_labels[i]
                print(f"  Dim {info['dim']:2d}: {info['controls']:20s} (r={info['correlation']:+.3f})")

            input("\nPress Enter to continue...")
            return True

        except Exception as e:
            print(f"\nERROR: Failed to initialize system.")
            print(f"Details: {e}")
            input("\nPress Enter to continue...")
            return False

    def generate_with_targets(self):
        """Generate airfoils with target features."""
        if not self._check_initialized():
            return

        print("\n" + "=" * 80)
        print("GENERATE AIRFOILS WITH TARGET FEATURES")
        print("=" * 80)

        try:
            # Get user input
            print("\nEnter target specifications:")
            thickness = float(input("  Target thickness (0.05-0.20, e.g., 0.12 for 12%): "))
            camber = float(input("  Target camber (0.00-0.08, e.g., 0.03 for 3%): "))
            num_samples = int(input("  Number of airfoils to generate (1-50): "))

            if not (0.05 <= thickness <= 0.20):
                print("\nWARNING: Thickness outside typical range (5-20%)")
            if not (0.0 <= camber <= 0.08):
                print("\nWARNING: Camber outside typical range (0-8%)")

            print(f"\nGenerating {num_samples} airfoils with:")
            print(f"  Thickness: {thickness:.1%}")
            print(f"  Camber: {camber:.1%}")

            # Generate
            airfoils, z = self.interpreter.generate_with_target_features(
                target_thickness=thickness,
                target_camber=camber,
                num_samples=num_samples
            )

            if airfoils is None:
                print("\nNo matching airfoils found. Try adjusting your targets.")
                input("\nPress Enter to continue...")
                return

            # Save and visualize
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = self.output_dir / f"generated_{timestamp}"
            output_folder.mkdir(exist_ok=True)

            print(f"\nSaving to: {output_folder}/")

            # Save each airfoil
            for i in range(len(airfoils)):
                coords = airfoils[i].numpy().reshape(200, 2)

                # Save .dat
                dat_file = output_folder / f"airfoil_{i+1:03d}.dat"
                with open(dat_file, 'w') as f:
                    f.write(f"Generated by DeepFoil - Target t/c={thickness:.1%}, camber={camber:.1%}\n")
                    for x, y in coords:
                        f.write(f"{x:.6f}  {y:.6f}\n")

                # Save .npy
                np.save(output_folder / f"airfoil_{i+1:03d}.npy", coords)

            # Create overview plot
            rows = (num_samples + 4) // 5
            fig, axes = plt.subplots(rows, 5, figsize=(20, rows*4))
            if rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()

            fig.suptitle(f'Generated Airfoils (Target: t/c={thickness:.1%}, camber={camber:.1%})',
                        fontsize=16, fontweight='bold')

            for i in range(len(airfoils)):
                coords = airfoils[i].numpy().reshape(200, 2)
                ax = axes[i]
                ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2)
                ax.set_aspect('equal')
                ax.set_title(f'Airfoil {i+1}', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.15, 0.15)

            for i in range(len(airfoils), len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(output_folder / 'overview.png', dpi=150, bbox_inches='tight')
            plt.close()

            # Save latent codes
            np.save(output_folder / 'latent_codes.npy', z.numpy())

            print(f"\nGenerated {len(airfoils)} airfoils successfully!")
            print(f"\nFiles saved:")
            print(f"  - {len(airfoils)} x .dat files")
            print(f"  - {len(airfoils)} x .npy files")
            print(f"  - overview.png")
            print(f"  - latent_codes.npy")

            input("\nPress Enter to continue...")

        except ValueError as e:
            print(f"\nERROR: Invalid input - {e}")
            input("\nPress Enter to continue...")
        except Exception as e:
            print(f"\nERROR: {e}")
            input("\nPress Enter to continue...")

    def explore_dimensions(self):
        """Explore dimension effects interactively."""
        if not self._check_initialized():
            return

        print("\n" + "=" * 80)
        print("EXPLORE DIMENSION EFFECTS")
        print("=" * 80)

        print("\nTop controllable dimensions:")
        for i in range(min(10, len(self.interpreter.dimension_labels))):
            info = self.interpreter.dimension_labels[i]
            print(f"  {i+1:2d}. Dim {info['dim']:2d}: {info['controls']:20s} (r={info['correlation']:+.3f})")

        try:
            choice = int(input("\nSelect dimension to explore (1-10): ")) - 1
            if choice < 0 or choice >= min(10, len(self.interpreter.dimension_labels)):
                print("Invalid choice.")
                input("\nPress Enter to continue...")
                return

            dim_info = self.interpreter.dimension_labels[choice]
            dimension = dim_info['dim']

            base_idx = int(input("Base airfoil index (0-1645, or press Enter for middle): ") or
                          str(len(self.interpreter.dataset) // 2))

            print(f"\nExploring Dimension {dimension} ({dim_info['controls']})")
            print("Generating variations...")

            fig = self.interpreter.manipulate_dimension(
                base_airfoil_idx=base_idx,
                dimension=dimension,
                strength=2.0,
                num_steps=7
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"dimension_{dimension}_exploration_{timestamp}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\nSaved: {output_file}")
            input("\nPress Enter to continue...")

        except ValueError as e:
            print(f"\nERROR: Invalid input - {e}")
            input("\nPress Enter to continue...")
        except Exception as e:
            print(f"\nERROR: {e}")
            input("\nPress Enter to continue...")

    def analyze_latent_space(self):
        """Display latent space analysis."""
        if not self._check_initialized():
            return

        print("\n" + "=" * 80)
        print("LATENT SPACE ANALYSIS")
        print("=" * 80)

        print("\nGenerating correlation visualization...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = self.output_dir / f"analysis_{timestamp}"
        output_folder.mkdir(exist_ok=True)

        self.interpreter.visualize_dimension_effects(str(output_folder))

        print(f"\nAnalysis complete! Saved to: {output_folder}/")
        print("\nFiles created:")
        print("  - dimension_correlations.png (heatmap)")
        print("  - dim_XX_effect.png (top 10 dimensions)")

        input("\nPress Enter to continue...")

    def batch_generate(self):
        """Generate a batch of diverse airfoils."""
        if not self._check_initialized():
            return

        print("\n" + "=" * 80)
        print("BATCH GENERATE AIRFOILS")
        print("=" * 80)

        try:
            num_samples = int(input("\nNumber of airfoils to generate (1-100): "))

            print("\nGeneration method:")
            print("  1. GMM sampling (best quality, diverse)")
            print("  2. Gaussian sampling (simple)")
            print("  3. Empirical sampling (safest)")
            method_choice = int(input("Choose method (1-3): "))

            methods = {1: 'gmm', 2: 'gaussian', 3: 'empirical'}
            method = methods.get(method_choice, 'gmm')

            print(f"\nGenerating {num_samples} airfoils using {method} method...")

            airfoils, z = self.generator.generate_batch(
                num_samples=num_samples,
                method=method,
                diversity=1.0
            )

            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = self.output_dir / f"batch_{timestamp}"
            output_folder.mkdir(exist_ok=True)

            for i in range(len(airfoils)):
                coords = airfoils[i].numpy().reshape(200, 2)

                # Save .dat
                dat_file = output_folder / f"airfoil_{i+1:03d}.dat"
                with open(dat_file, 'w') as f:
                    f.write(f"Generated by DeepFoil - {method} method\n")
                    for x, y in coords:
                        f.write(f"{x:.6f}  {y:.6f}\n")

                np.save(output_folder / f"airfoil_{i+1:03d}.npy", coords)

            # Overview
            rows = (num_samples + 4) // 5
            fig, axes = plt.subplots(rows, 5, figsize=(20, rows*4))
            if rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()

            fig.suptitle(f'{num_samples} Generated Airfoils ({method} method)',
                        fontsize=16, fontweight='bold')

            for i in range(len(airfoils)):
                coords = airfoils[i].numpy().reshape(200, 2)
                ax = axes[i]
                ax.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=2)
                ax.set_aspect('equal')
                ax.set_title(f'Airfoil {i+1}', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.15, 0.15)

            for i in range(len(airfoils), len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(output_folder / 'overview.png', dpi=150, bbox_inches='tight')
            plt.close()

            np.save(output_folder / 'latent_codes.npy', z.numpy())

            print(f"\nGenerated {len(airfoils)} airfoils successfully!")
            print(f"Saved to: {output_folder}/")

            input("\nPress Enter to continue...")

        except Exception as e:
            print(f"\nERROR: {e}")
            input("\nPress Enter to continue...")

    def guided_workflow(self):
        """Guided design workflow."""
        if not self._check_initialized():
            return

        print("\n" + "=" * 80)
        print("GUIDED DESIGN WORKFLOW")
        print("=" * 80)

        print("\nThis workflow will guide you through a complete airfoil design process.")
        print("\nStep 1: Define Requirements")
        print("-" * 40)

        try:
            print("\nWhat is your application?")
            application = input("  (e.g., 'UAV surveillance drone', 'Wind turbine'): ")

            print("\nDesign requirements:")
            min_thickness = float(input("  Minimum thickness (e.g., 0.10 for 10%): "))
            max_thickness = float(input("  Maximum thickness (e.g., 0.14 for 14%): "))
            target_camber = float(input("  Target camber (e.g., 0.03 for 3%): "))

            print("\nStep 2: Generate Initial Candidates")
            print("-" * 40)

            target_thickness = (min_thickness + max_thickness) / 2
            print(f"\nGenerating candidates with thickness={target_thickness:.1%}, camber={target_camber:.1%}...")

            airfoils, z = self.interpreter.generate_with_target_features(
                target_thickness=target_thickness,
                target_camber=target_camber,
                num_samples=10
            )

            if airfoils is None:
                print("\nNo matching airfoils found.")
                input("\nPress Enter to continue...")
                return

            # Filter by thickness range
            valid_airfoils = []
            valid_z = []
            for i in range(len(airfoils)):
                coords = airfoils[i].numpy().reshape(200, 2)
                upper = coords[:100]
                lower = coords[100:]
                thickness = np.max(upper[:, 1] - lower[:, 1])

                if min_thickness <= thickness <= max_thickness:
                    valid_airfoils.append(airfoils[i])
                    valid_z.append(z[i])

            print(f"\n{len(valid_airfoils)}/{len(airfoils)} candidates meet thickness requirements")

            if len(valid_airfoils) == 0:
                print("\nNo candidates met requirements. Try adjusting your targets.")
                input("\nPress Enter to continue...")
                return

            print("\nStep 3: Save Final Designs")
            print("-" * 40)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = self.output_dir / f"workflow_{timestamp}"
            output_folder.mkdir(exist_ok=True)

            # Save designs
            for i, airfoil in enumerate(valid_airfoils):
                coords = airfoil.numpy().reshape(200, 2)

                dat_file = output_folder / f"design_{i+1}.dat"
                with open(dat_file, 'w') as f:
                    f.write(f"Generated by DeepFoil for: {application}\n")
                    f.write(f"Requirements: t/c={min_thickness:.1%}-{max_thickness:.1%}, camber={target_camber:.1%}\n")
                    for x, y in coords:
                        f.write(f"{x:.6f}  {y:.6f}\n")

                np.save(output_folder / f"design_{i+1}.npy", coords)

            # Create report
            report_file = output_folder / "DESIGN_REPORT.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("DEEPFOIL DESIGN REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Application: {application}\n\n")
                f.write("REQUIREMENTS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Thickness: {min_thickness:.1%} - {max_thickness:.1%}\n")
                f.write(f"Camber: {target_camber:.1%}\n\n")
                f.write("GENERATED DESIGNS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Number of designs: {len(valid_airfoils)}\n\n")

                for i, airfoil in enumerate(valid_airfoils):
                    coords = airfoil.numpy().reshape(200, 2)
                    upper = coords[:100]
                    lower = coords[100:]
                    thickness = np.max(upper[:, 1] - lower[:, 1])
                    camber_line = (upper[:, 1] + lower[:, 1]) / 2
                    camber = np.max(np.abs(camber_line))

                    f.write(f"\nDesign {i+1}:\n")
                    f.write(f"  File: design_{i+1}.dat\n")
                    f.write(f"  Thickness: {thickness:.2%}\n")
                    f.write(f"  Camber: {camber:.2%}\n")

            # Visualization
            fig, axes = plt.subplots(len(valid_airfoils), 1, figsize=(12, len(valid_airfoils)*3))
            if len(valid_airfoils) == 1:
                axes = [axes]

            fig.suptitle(f'Final Designs for: {application}', fontsize=14, fontweight='bold')

            for i, airfoil in enumerate(valid_airfoils):
                coords = airfoil.numpy().reshape(200, 2)
                ax = axes[i]
                ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2)
                ax.fill_between(coords[:, 0], coords[:, 1], alpha=0.3)
                ax.set_aspect('equal')
                ax.set_title(f'Design {i+1}', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.15, 0.15)
                ax.set_xlabel('x/c')
                ax.set_ylabel('y/c')

            plt.tight_layout()
            plt.savefig(output_folder / 'final_designs.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\nWorkflow complete!")
            print(f"Saved to: {output_folder}/")
            print(f"\nGenerated:")
            print(f"  - {len(valid_airfoils)} design files (.dat and .npy)")
            print(f"  - DESIGN_REPORT.txt")
            print(f"  - final_designs.png")

            input("\nPress Enter to continue...")

        except Exception as e:
            print(f"\nERROR: {e}")
            input("\nPress Enter to continue...")

    def settings_menu(self):
        """Settings menu."""
        print("\n" + "=" * 80)
        print("SETTINGS")
        print("=" * 80)
        print(f"\nCurrent output directory: {self.output_dir.absolute()}")
        print("\nNo configurable settings at this time.")
        input("\nPress Enter to continue...")

    def help_menu(self):
        """Help and documentation."""
        print("\n" + "=" * 80)
        print("HELP & DOCUMENTATION")
        print("=" * 80)

        print("\n1. SYSTEM OVERVIEW")
        print("-" * 40)
        print("DeepFoil uses an autoencoder to learn a compressed representation")
        print("of airfoil geometry. This enables:")
        print("  - Rapid generation of new airfoil designs")
        print("  - Control over specific geometric features (thickness, camber)")
        print("  - Exploration of design trade-offs")

        print("\n2. TYPICAL WORKFLOW")
        print("-" * 40)
        print("  1. Initialize System (loads model and analyzes latent space)")
        print("  2. Generate airfoils with your target specifications")
        print("  3. Optionally explore dimension effects to understand trade-offs")
        print("  4. Export designs for CFD analysis")

        print("\n3. OUTPUT FILES")
        print("-" * 40)
        print("  .dat files: Industry-standard airfoil coordinate format")
        print("  .npy files: NumPy binary format for Python")
        print("  .png files: Visualizations")
        print("  latent_codes.npy: Latent space representations (for reproduction)")

        print("\n4. KEY CONCEPTS")
        print("-" * 40)
        print("  Latent Space: Compressed representation of airfoil geometry")
        print("  Dimension: A single number in the latent representation")
        print("  Correlation: How strongly a dimension controls a feature")

        print("\n5. LIMITATIONS")
        print("-" * 40)
        print("  - Generated airfoils are geometric only (not optimized for performance)")
        print("  - Must validate with CFD or wind tunnel testing")
        print("  - Limited to single-element airfoils")
        print("  - Based on training data from UIUC database")

        print("\n6. SUPPORT")
        print("-" * 40)
        print("  README.md contains detailed documentation")
        print("  Check examples/ folder for tutorials")

        input("\nPress Enter to continue...")

    def _check_initialized(self):
        """Check if system is initialized."""
        if self.interpreter is None or self.generator is None:
            print("\nERROR: System not initialized.")
            print("Please select option 1 to initialize the system first.")
            input("\nPress Enter to continue...")
            return False
        return True

    def run(self):
        """Main application loop."""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_menu()

            try:
                choice = input("\nEnter your choice (1-9): ").strip()

                if choice == '1':
                    self.initialize_system()
                elif choice == '2':
                    self.generate_with_targets()
                elif choice == '3':
                    self.explore_dimensions()
                elif choice == '4':
                    self.analyze_latent_space()
                elif choice == '5':
                    self.batch_generate()
                elif choice == '6':
                    self.guided_workflow()
                elif choice == '7':
                    self.settings_menu()
                elif choice == '8':
                    self.help_menu()
                elif choice == '9':
                    print("\nThank you for using DeepFoil!")
                    break
                else:
                    print("\nInvalid choice. Please try again.")
                    input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except EOFError:
                print("\n\nError: Interactive input not available.")
                print("DeepFoil requires an interactive terminal to run the main interface.")
                print("\nAlternatives:")
                print("  1. Run from a real terminal/command prompt")
                print("  2. Use the example scripts: python examples/simple_generation.py")
                print("  3. Use the programmatic API (see QUICKSTART.md)")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                try:
                    input("\nPress Enter to continue...")
                except EOFError:
                    break


if __name__ == "__main__":
    app = DeepFoilInterface()
    app.run()
