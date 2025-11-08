"""
create_dataset.py

Creates a PyTorch Dataset from parsed airfoils for machine learning.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import sys
from pathlib import Path

# Add project root to path so imports work when running directly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.parse_airfoils import load_all_airfoils


class AirfoilDataset(Dataset):
    """
    PyTorch Dataset for airfoil coordinates.
    Normalizes all airfoils to fixed number of points.
    """
    
    def __init__(self, airfoils, n_points=200):
        """
        Initialize dataset from list of Airfoil objects.
        
        Args:
            airfoils: List of Airfoil objects from parse_airfoils.py
            n_points: Number of points to normalize each airfoil to
        """
        self.n_points = n_points
        self.data = []
        
        print(f"Processing {len(airfoils)} airfoils...")
        
        failed_count = 0
        
        for i, airfoil in enumerate(airfoils):
            try:
                # Normalize to fixed number of points
                coords = airfoil.normalize_points(n_points)
                
                # Check if normalization actually worked
                if coords is None or len(coords) != n_points:
                    failed_count += 1
                    continue
                
                # Flatten to 1D: [x1, y1, x2, y2, ..., xn, yn]
                coords_flat = coords.flatten()
                
                # Verify shape is correct
                if coords_flat.shape[0] != n_points * 2:
                    failed_count += 1
                    continue
                
                # Check for NaN or inf values
                if np.any(np.isnan(coords_flat)) or np.any(np.isinf(coords_flat)):
                    failed_count += 1
                    continue
                
                # Store
                self.data.append(coords_flat)
                
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # Only print first 5 errors
                    print(f"  Failed: {airfoil.name} - {e}")
                continue
        
        # Convert to numpy array for efficiency
        self.data = np.array(self.data)
        
        print(f"[OK] Dataset created: {len(self.data)} valid airfoils")
        print(f"[FAIL] Failed to process: {failed_count} airfoils")
        if len(self.data) > 0:
            print(f"   Shape of each airfoil: {self.data[0].shape}")
    
    def __len__(self):
        """
        Return the number of airfoils in the dataset.
        Required by PyTorch.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Return a single airfoil at the given index as a PyTorch tensor.
        Required by PyTorch.
        
        Args:
            idx: Index of airfoil to return (0 to len-1)
            
        Returns:
            torch.Tensor: Flattened airfoil coordinates, shape [400]
        """
        # Convert numpy array to PyTorch tensor
        return torch.FloatTensor(self.data[idx])
    
    def save(self, filepath):
        """
        Save dataset to file for later use.
        
        Args:
            filepath: Path to save the dataset (e.g., 'data/processed/airfoil_dataset.pkl')
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'data': self.data,
                'n_points': self.n_points
            }, f)
        print(f"[OK] Saved dataset to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load dataset from file.
        
        Args:
            filepath: Path to the saved dataset file
            
        Returns:
            AirfoilDataset: Loaded dataset
        """
        with open(filepath, 'rb') as f:
            saved = pickle.load(f)
        
        # Create empty dataset object
        dataset = cls.__new__(cls)
        dataset.data = saved['data']
        dataset.n_points = saved['n_points']
        
        print(f"[OK] Loaded dataset from {filepath}")
        print(f"   Contains {len(dataset.data)} airfoils")
        print(f"   Shape per airfoil: {dataset.data[0].shape}")
        
        return dataset


def main():
    """
    Main script to create and save the dataset.
    Run this to process all airfoils and save for training.
    """
    print("="*60)
    print("Creating Airfoil Dataset for Machine Learning")
    print("="*60)
    
    # Step 1: Load all airfoils
    print("\n1. Loading airfoils from UIUC database...")
    airfoils = load_all_airfoils("data/raw/uiuc")
    
    if not airfoils:
        print("❌ ERROR: No airfoils loaded!")
        print("   Make sure data/raw/uiuc/ contains .dat files")
        return
    
    # Step 2: Create dataset
    print("\n2. Creating PyTorch dataset...")
    dataset = AirfoilDataset(airfoils, n_points=200)
    
    if len(dataset) == 0:
        print("❌ ERROR: No valid airfoils in dataset!")
        return
    
    # Step 3: Test the dataset
    print("\n3. Testing dataset...")
    print(f"   Total airfoils: {len(dataset)}")
    print(f"   First airfoil shape: {dataset[0].shape}")
    print(f"   First airfoil type: {type(dataset[0])}")
    print(f"   Data range: [{dataset[0].min():.6f}, {dataset[0].max():.6f}]")
    
    # Step 4: Visualize a few examples
    print("\n4. Creating visualization...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(min(6, len(dataset))):
        # Get airfoil and reshape back to (200, 2)
        coords_flat = dataset[i].numpy()
        coords = coords_flat.reshape(-1, 2)
        
        # Plot
        ax = axes[i]
        ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.set_title(f'Airfoil {i}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/dataset_samples.png', dpi=150)
    print("   [OK] Saved visualization to outputs/plots/dataset_samples.png")
    
    # Step 5: Save dataset
    print("\n5. Saving dataset...")
    dataset.save("data/processed/airfoil_dataset.pkl")
    
    # Step 6: Test loading
    print("\n6. Testing load functionality...")
    loaded_dataset = AirfoilDataset.load("data/processed/airfoil_dataset.pkl")
    
    # Verify it matches
    assert len(loaded_dataset) == len(dataset), "Loaded dataset length mismatch!"
    assert np.allclose(loaded_dataset[0].numpy(), dataset[0].numpy()), "Loaded data mismatch!"
    print("   [OK] Load test passed!")
    
    print("\n" + "="*60)
    print("[SUCCESS] Dataset is ready for training!")
    print("="*60)
    print(f"\nDataset location: data/processed/airfoil_dataset.pkl")
    print(f"Total airfoils: {len(dataset)}")
    print(f"Points per airfoil: 200")
    print(f"Input dimension: 400 (flattened)")
    print("\nNext steps:")
    print("  - Week 2 Wednesday: Validate dataset")
    print("  - Week 2 Thursday: Create train/val split")
    print("  - Week 3: Build VAE model")


if __name__ == "__main__":
    main()