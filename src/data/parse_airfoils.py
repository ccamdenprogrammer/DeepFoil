import numpy as np
import os
import glob
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class Airfoil:
    #Class to represent an airfoil with methods for normalization, feature extraction, and visualization.
    
    def __init__(self, name, coordinates):
        #initialize airfoil with name and coordinates
        self.name = name
        self.coordinates = np.array(coordinates)
        
    def normalize_points(self, n_points=200):
   
        coords = self.coordinates
    
        # Find leading edge (minimum x value)
        le_idx = np.argmin(coords[:, 0])
    
        # Split airfoil coords into upper and lower surfaces  
        upper = coords[:le_idx+1]
        lower = coords[le_idx+1:]  # Start AFTER leading edge
    
        # Handle edge case
        if len(lower) < 2:
            lower = coords[le_idx:]
    
        # Resample each surface
        upper_resampled = self._resample_curve(upper, n_points // 2)
        lower_resampled = self._resample_curve(lower, n_points // 2)
    
        # Combine surfaces
        normalized = np.vstack([upper_resampled, lower_resampled])
    
        return normalized
    
    def _resample_curve(self, points, n_points):
        #here we resample a curve defined by 'points' to have 'n_points' using interpolation

        #|
        #|
        #|
        #V


        if len(points) < 2:
            return points
        
        # Calculate cumulative arc length
        differences = np.diff(points, axis=0)
        distances = np.sqrt(np.sum(differences**2, axis=1))
        cumulative_distance = np.concatenate([[0], np.cumsum(distances)])
        
        if cumulative_distance[-1] == 0:
            return np.repeat(points[[0]], n_points, axis=0)
        
        # Interpolate x and y separately
        try:
            fx = interp1d(cumulative_distance, points[:, 0], kind='cubic', 
                         bounds_error=False, fill_value='extrapolate')
            fy = interp1d(cumulative_distance, points[:, 1], kind='cubic',
                         bounds_error=False, fill_value='extrapolate')
        except:
            fx = interp1d(cumulative_distance, points[:, 0], kind='linear',
                         bounds_error=False, fill_value='extrapolate')
            fy = interp1d(cumulative_distance, points[:, 1], kind='linear',
                         bounds_error=False, fill_value='extrapolate')
        
        # Sample at evenly spaced distances
        new_distances = np.linspace(0, cumulative_distance[-1], n_points)
        new_x = fx(new_distances)
        new_y = fy(new_distances)
        
        return np.column_stack([new_x, new_y])
    
    def get_geometry_features(self):
        #Extract key geometric features of the airfoil, such as maximum thickness and its location.
        coords = self.coordinates
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Find array of unique x values
        x_unique = np.unique(x)
        
        # Calculate thickness at each x
        thicknesses = []
        for xi in x_unique:
            y_at_x = y[x == xi]
            if len(y_at_x) >= 2:
                thickness = np.max(y_at_x) - np.min(y_at_x)
                thicknesses.append((xi, thickness))
        
        # Find maximum thickness
        if thicknesses:
            max_thick = max(thicknesses, key=lambda t: t[1])
            max_thickness = max_thick[1]
            max_thickness_loc = max_thick[0]
        else:
            max_thickness = 0
            max_thickness_loc = 0
        
        return {
            'max_thickness': max_thickness,
            'max_thickness_location': max_thickness_loc,
        }
    
    def plot(self, ax=None, show_features=False):
       #Plot and visualize the airfoil shape.
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(self.coordinates[:, 0], self.coordinates[:, 1], 'b-', linewidth=2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        
        return ax


def parse_airfoil_file(filepath):
    #This function actually gets the airfoil data from a .dat file

    #open the file and read lines
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except:
        return None
    
    if len(lines) < 2:
        return None
    
    # The first line in the file is the name of the airfoil
    name = lines[0].strip()
    
    # Parse coordinates
    coordinates = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                coordinates.append([x, y])
            except ValueError:
                continue
    
    if len(coordinates) < 10:
        return None
    
    return Airfoil(name, np.array(coordinates))


def load_all_airfoils(directory="data/raw/uiuc", verbose=True):
    #Here we are loading all airfoils from our dataset directory

    #Initialize airfoil list and failure counter
    airfoils = []
    failed = 0
    
    #Find all .dat files in the specified directory
    dat_files = glob.glob(os.path.join(directory, "*.dat"))
    
    #Verbose output of number of files found
    if verbose:
        print(f"Found {len(dat_files)} .dat files")
    
    for filepath in dat_files:
        airfoil = parse_airfoil_file(filepath)
        if airfoil is not None:
            airfoils.append(airfoil)
        else:
            failed += 1
    
    if verbose:
        print(f"Successfully loaded: {len(airfoils)}")
        print(f"Failed to load: {failed}")
    
    return airfoils


def validate_airfoil(airfoil, checks=None):
    """
    Validate that an airfoil has reasonable geometry.
    """
    if checks is None:
        checks = ['closed', 'range']
    
    errors = []
    
    if 'closed' in checks:
        # Check if trailing edge is reasonably closed
        first_point = airfoil.coordinates[0]
        last_point = airfoil.coordinates[-1]
        distance = np.linalg.norm(first_point - last_point)
        if distance > 0.1:
            errors.append(f"Trailing edge not closed (distance: {distance:.4f})")
    
    if 'range' in checks:
        # Check if coordinates are in reasonable range
        x = airfoil.coordinates[:, 0]
        y = airfoil.coordinates[:, 1]
        if np.min(x) < -0.1 or np.max(x) > 1.1:
            errors.append(f"X coordinates out of range: [{np.min(x):.3f}, {np.max(x):.3f}]")
        if np.abs(np.max(y)) > 1.0 or np.abs(np.min(y)) > 1.0:
            errors.append(f"Y coordinates out of range: [{np.min(y):.3f}, {np.max(y):.3f}]")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def compare_airfoils(airfoil1, airfoil2, plot=True):
    """
    Compare two airfoils visually and compute geometric differences.
    """
    # Normalize both to same number of points
    coords1 = airfoil1.normalize_points(200)
    coords2 = airfoil2.normalize_points(200)
    
    # Calculate point-wise differences
    differences = coords1 - coords2
    rms_diff = np.sqrt(np.mean(differences**2))
    
    results = {
        'rms_difference': rms_diff,
        'max_difference': np.max(np.abs(differences)),
    }
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(coords1[:, 0], coords1[:, 1], 'b-', label=airfoil1.name, linewidth=2)
        ax.plot(coords2[:, 0], coords2[:, 1], 'r--', label=airfoil2.name, linewidth=2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.set_title('Airfoil Comparison')
        plt.tight_layout()
    
    return results


def main():
    """
    Main function for testing the parser.
    """
    print("="*60)
    print("Airfoil Parser - Test Run")
    print("="*60)
    
    # Load all airfoils
    print("\n1. Loading airfoils from UIUC database...")
    airfoils = load_all_airfoils("data/raw/uiuc")
    
    if not airfoils:
        print("ERROR: No airfoils loaded. Check that data/raw/uiuc/ contains .dat files")
        return
    
    print(f"✓ Successfully loaded {len(airfoils)} airfoils")
    
    # Show some examples
    print("\n2. Example airfoils:")
    for i, af in enumerate(airfoils[:5]):
        print(f"   - {af.name}: {len(af.coordinates)} points")
    
    # Test normalization
    print("\n3. Testing normalization...")
    test_airfoil = airfoils[0]
    print(f"   Original: {len(test_airfoil.coordinates)} points")
    normalized = test_airfoil.normalize_points(200)
    print(f"   Normalized: {len(normalized)} points")
    
    # Test geometry extraction
    print("\n4. Testing geometry feature extraction...")
    features = test_airfoil.get_geometry_features()
    print(f"   Features extracted for {test_airfoil.name}")
    for key, value in features.items():
        print(f"     - {key}: {value:.4f}")
    
    # Visualize examples
    print("\n5. Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(min(6, len(airfoils))):
        af = airfoils[i]
        af.plot(ax=axes[i])
        axes[i].set_title(af.name, fontsize=10)
    
    plt.tight_layout()
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig('outputs/plots/parsed_airfoils_test.png', dpi=150)
    print("   ✓ Saved visualization to outputs/plots/parsed_airfoils_test.png")
    
    # Test validation
    print("\n6. Testing airfoil validation...")
    for i in range(min(3, len(airfoils))):
        is_valid, errors = validate_airfoil(airfoils[i])
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"   {status}: {airfoils[i].name}")
        if errors:
            for error in errors:
                print(f"      - {error}")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


if __name__ == "__main__":
    main()