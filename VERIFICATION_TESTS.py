"""
Comprehensive Verification Tests for DeepFoil Minimal Distribution

This script tests all core functionality to ensure the distribution works correctly.
"""

import sys
from pathlib import Path
import numpy as np
import torch

def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*80)
    print("TEST 1: Import Verification")
    print("="*80)

    try:
        from latent_interpreter import LatentInterpreter
        from generate_airfoils import AirfoilGenerator
        from src.models.airfoil_ae import AirfoilAE
        from src.data.create_dataset import AirfoilDataset
        print("[PASS] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False

def test_file_existence():
    """Test that required files exist."""
    print("\n" + "="*80)
    print("TEST 2: Required Files Existence")
    print("="*80)

    required_files = [
        "models/airfoil_ae/best_model.pth",
        "data/processed/airfoil_dataset.pkl",
        "latent_interpreter.py",
        "generate_airfoils.py",
        "deepfoil_interface.py",
        "requirements.txt",
        "README.md",
        "QUICK_DEMO.py",
        "LICENSE",
    ]

    all_exist = True
    for file_path in required_files:
        exists = Path(file_path).exists()
        status = "[PASS]" if exists else "[FAIL]"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False

    return all_exist

def test_latent_interpreter():
    """Test LatentInterpreter functionality."""
    print("\n" + "="*80)
    print("TEST 3: LatentInterpreter Functionality")
    print("="*80)

    try:
        from latent_interpreter import LatentInterpreter

        print("  Initializing LatentInterpreter...")
        interpreter = LatentInterpreter()

        print(f"  Latent dimension: {interpreter.latent_dim}")
        print(f"  Dataset size: {len(interpreter.dataset)}")

        # Test generation
        print("  Testing generation with target features...")
        airfoils, z = interpreter.generate_with_target_features(
            target_thickness=0.12,
            target_camber=0.03,
            num_samples=3
        )

        if airfoils is None:
            print("[FAIL] No airfoils generated")
            return False

        print(f"  Generated {len(airfoils)} airfoils")

        # Verify shape
        coords = airfoils[0].numpy().reshape(200, 2)
        assert coords.shape == (200, 2), "Invalid airfoil shape"

        print("[PASS] LatentInterpreter works correctly")
        return True

    except Exception as e:
        print(f"[FAIL] LatentInterpreter error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_airfoil_generator():
    """Test AirfoilGenerator functionality."""
    print("\n" + "="*80)
    print("TEST 4: AirfoilGenerator Functionality")
    print("="*80)

    try:
        from generate_airfoils import AirfoilGenerator

        print("  Initializing AirfoilGenerator...")
        generator = AirfoilGenerator()

        # Test GMM generation
        print("  Testing GMM batch generation...")
        airfoils, z = generator.generate_batch(
            num_samples=5,
            method='gmm',
            diversity=1.0
        )

        print(f"  Generated {len(airfoils)} airfoils")

        # Verify shape
        coords = airfoils[0].numpy().reshape(200, 2)
        assert coords.shape == (200, 2), "Invalid airfoil shape"

        print("[PASS] AirfoilGenerator works correctly")
        return True

    except Exception as e:
        print(f"[FAIL] AirfoilGenerator error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test that the model loads correctly."""
    print("\n" + "="*80)
    print("TEST 5: Model Loading")
    print("="*80)

    try:
        from src.models.airfoil_ae import AirfoilAE

        print("  Loading model checkpoint...")
        checkpoint = torch.load(
            "models/airfoil_ae/best_model.pth",
            map_location='cpu',
            weights_only=False
        )

        latent_dim = checkpoint.get('latent_dim', 24)
        print(f"  Latent dimension: {latent_dim}")

        model = AirfoilAE(latent_dim=latent_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("  Testing model forward pass...")
        test_input = torch.randn(1, 400)
        with torch.no_grad():
            reconstructed, latent = model(test_input)

        assert reconstructed.shape == (1, 400), "Invalid output shape"
        assert latent.shape == (1, latent_dim), "Invalid latent shape"

        print("[PASS] Model loads and runs correctly")
        return True

    except Exception as e:
        print(f"[FAIL] Model loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test that the dataset loads correctly."""
    print("\n" + "="*80)
    print("TEST 6: Dataset Loading")
    print("="*80)

    try:
        import pickle

        print("  Loading pre-built dataset...")
        with open("data/processed/airfoil_dataset.pkl", 'rb') as f:
            dataset_dict = pickle.load(f)

        airfoils = dataset_dict['data']
        print(f"  Dataset size: {len(airfoils)}")
        print(f"  Points per airfoil: {dataset_dict['n_points']}")

        # Test getting an item
        airfoil = airfoils[0]
        print(f"  Airfoil shape: {airfoil.shape}")

        assert airfoil.shape == (400,), "Invalid airfoil shape"
        assert len(airfoils) > 1000, "Dataset too small"

        print("[PASS] Dataset loads correctly")
        return True

    except Exception as e:
        print(f"[FAIL] Dataset loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_coordinate_format():
    """Test airfoil coordinate format."""
    print("\n" + "="*80)
    print("TEST 7: Coordinate Format Validation")
    print("="*80)

    try:
        from latent_interpreter import LatentInterpreter

        interpreter = LatentInterpreter()
        airfoils, _ = interpreter.generate_with_target_features(
            target_thickness=0.12,
            target_camber=0.03,
            num_samples=1
        )

        if airfoils is None:
            print("[FAIL] Could not generate airfoil for testing")
            return False

        coords = airfoils[0].numpy().reshape(200, 2)

        # Verify x is in reasonable range (allow small overshoot)
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        print(f"  X range: [{x_min:.4f}, {x_max:.4f}]")
        assert -0.01 <= x_min and x_max <= 1.05, "X coordinates out of expected range"

        # Verify y is reasonable
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        print(f"  Y range: [{y_min:.4f}, {y_max:.4f}]")
        assert -0.3 < y_min and y_max < 0.3, "Y coordinates out of expected range"

        # Verify trailing edge closure
        te_gap = np.abs(coords[0, 1] - coords[-1, 1])
        print(f"  Trailing edge gap: {te_gap:.6f}")

        print("[PASS] Coordinate format is valid")
        return True

    except Exception as e:
        print(f"[FAIL] Coordinate format error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_examples_exist():
    """Test that example scripts exist."""
    print("\n" + "="*80)
    print("TEST 8: Example Scripts Existence")
    print("="*80)

    examples = [
        "examples/simple_generation.py",
        "examples/batch_processing.py",
        "examples/custom_workflow.py"
    ]

    all_exist = True
    for example in examples:
        exists = Path(example).exists()
        status = "[PASS]" if exists else "[FAIL]"
        print(f"{status} {example}")
        if not exists:
            all_exist = False

    return all_exist

def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*20 + "DeepFoil Verification Tests" + " "*31 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    tests = [
        ("Import Verification", test_imports),
        ("Required Files", test_file_existence),
        ("LatentInterpreter", test_latent_interpreter),
        ("AirfoilGenerator", test_airfoil_generator),
        ("Model Loading", test_model_loading),
        ("Dataset Loading", test_dataset_loading),
        ("Coordinate Format", test_coordinate_format),
        ("Example Scripts", test_examples_exist),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{name}' crashed: {e}")
            results.append((name, False))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    print("="*80)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed! Distribution is valid.")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
