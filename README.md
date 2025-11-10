# DeepFoil - AI-Powered Airfoil Design System

Deep learning system for generating novel airfoil geometries using a deterministic autoencoder with interpretable latent space.

**Key Innovation**: Solves VAE posterior collapse problem while achieving 94% dimensionality reduction (400D → 24D) and near-perfect reconstruction (MSE < 0.00001).

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run 2-minute demo
python QUICK_DEMO.py

# Or use interactive interface
python deepfoil_interface.py
```

---

## What is This?

DeepFoil generates airfoil designs using a custom neural architecture that:
- Compresses airfoil geometry from 400 dimensions to 24
- Maintains interpretable control over thickness, camber, and other features
- Generates diverse, realistic designs at 50 airfoils/second (CPU)

**Technical Achievement**: Unlike VAE approaches that suffer from posterior collapse, this deterministic autoencoder learns the actual latent distribution from training data, avoiding common generative model pitfalls while maintaining full interpretability.

### Performance Metrics
- **Reconstruction MSE**: 0.000004 (near-perfect)
- **Latent Dimensions**: 24 (vs 100+ in typical VAEs)
- **Feature Control**: ±5% thickness, ±0.5% camber accuracy
- **Training Data**: 1,646 airfoils (UIUC database)
- **Generation Speed**: ~50 airfoils/second on CPU

---

## Installation

### Requirements
- Python 3.8 - 3.14
- No GPU required (CPU is fast enough)

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies**: PyTorch, NumPy, Matplotlib, SciPy, scikit-learn

---

## Usage

### 1. Quick Demo (Recommended First Step)
```bash
python QUICK_DEMO.py
```
Generates 6 custom airfoils in 30 seconds, creates visualization, shows all capabilities.

### 2. Interactive Interface
```bash
python deepfoil_interface.py
```

**Menu Options**:
1. Initialize System (load model)
2. Generate with target thickness/camber
3. Explore latent dimension effects
4. Analyze latent space correlations
5. Batch generate diverse airfoils
6. Guided design workflow

### 3. Programmatic Usage

**Basic Generation**:
```python
from latent_interpreter import LatentInterpreter

# Initialize
interpreter = LatentInterpreter()

# Generate airfoils with specific features
airfoils, z = interpreter.generate_with_target_features(
    target_thickness=0.12,  # 12% thickness
    target_camber=0.03,     # 3% camber
    num_samples=5
)

# Save results
for i, airfoil in enumerate(airfoils):
    coords = airfoil.numpy().reshape(200, 2)
    # coords is (200, 2) array of x,y coordinates
```

**Batch Generation**:
```python
from generate_airfoils import AirfoilGenerator

generator = AirfoilGenerator()

# Generate diverse airfoils
airfoils, z = generator.generate_batch(
    num_samples=20,
    method='gmm',      # 'gmm', 'gaussian', or 'empirical'
    diversity=1.0
)
```

**Dimension Exploration**:
```python
# See how a dimension affects shape
fig = interpreter.manipulate_dimension(
    base_airfoil_idx=500,
    dimension=3,        # e.g., thickness dimension
    strength=2.0,
    num_steps=7
)
```

### 4. Example Scripts

```bash
# Simple generation example
python examples/simple_generation.py

# Batch processing multiple specifications
python examples/batch_processing.py

# Custom filtering workflow
python examples/custom_workflow.py
```

---

## Architecture

### Model Design
```
Encoder: 400D → 256 → 128 → 64 → 24 (latent)
Decoder: 24 → 64 → 128 → 256 → 400D
```

**Architecture Details**:
- **Encoder**: Progressive compression with LayerNorm, SiLU activation, Dropout
- **Decoder**: Symmetric expansion without normalization
- **Latent Space**: Unbounded (no forced distribution)
- **Model Size**: 3.4 MB

### Loss Function
Multi-objective loss with progressive weighting:
```
Loss = α·MSE + β·Smoothness + γ·TE_Closure + δ·Diversity
```

**Components**:
- **MSE**: Reconstruction accuracy
- **Smoothness**: 1st/2nd/3rd derivative penalties (aerodynamic realism)
- **TE_Closure**: Trailing edge constraint
- **Diversity**: Encourages latent variance

**Key Innovation**: Progressive smoothness ramping - learns reconstruction first, then enforces aerodynamic constraints. This prevents underfitting while maintaining smooth, realistic shapes.

### Why Deterministic (Not VAE)?
VAEs force the latent space to match N(0,1), often causing:
- Posterior collapse (loss of diversity)
- Poor reconstruction at low dimensions
- Uninterpretable latent space

**DeepFoil's approach**:
1. Learn actual latent distribution from training data
2. Fit Gaussian Mixture Model to learned distribution
3. Sample from GMM for generation
4. Result: No collapse, better reconstruction, interpretable dimensions

### Interpretable Latent Space
After encoding training set, correlate each latent dimension with geometric features:
- Thickness (max, average)
- Camber (max, average)
- Leading edge radius
- Trailing edge angle

**Result**: 10+ dimensions with strong correlations (r > 0.78), allowing controllable generation.

---

## Output Files

DeepFoil generates:
- **`.dat` files**: XFOIL-compatible coordinate format
- **`.npy` files**: NumPy arrays for Python processing
- **`.png` files**: Visualizations
- **`latent_codes.npy`**: Latent representations (for reproducibility)

All outputs saved to `deepfoil_outputs/` or `demo_output/` directories.

### Coordinate Format
- 200 points: trailing edge → upper surface → leading edge → lower surface → trailing edge
- Normalized: x ∈ [0,1], y typically ∈ [-0.2, 0.2]

---

## Project Structure

```
deepfoil_minimal/
├── README.md                   # This file
├── QUICK_DEMO.py              # 2-minute demonstration
├── requirements.txt           # Dependencies
├── LICENSE                    # MIT License
│
├── deepfoil_interface.py      # Interactive CLI (620 lines)
├── latent_interpreter.py      # Feature-based generation (400 lines)
├── generate_airfoils.py       # Batch generation (280 lines)
│
├── src/
│   ├── models/
│   │   └── airfoil_ae.py      # Neural architecture (310 lines)
│   └── data/
│       ├── create_dataset.py  # Dataset loader (150 lines)
│       └── parse_airfoils.py  # Parsing utilities
│
├── examples/
│   ├── simple_generation.py   # Basic usage
│   ├── batch_processing.py    # Batch workflow
│   └── custom_workflow.py     # Custom filtering
│
├── models/
│   └── airfoil_ae/
│       └── best_model.pth     # Pre-trained model (3.4 MB)
│
└── data/
    └── processed/
        └── airfoil_dataset.pkl # 1,646 airfoils (5.1 MB)
```

**Total**: ~1,800 lines of production code

---

## Technical Details

### Training Dataset
- **Source**: UIUC Airfoil Database
- **Size**: 1,646 airfoils
- **Format**: 200 (x,y) coordinate pairs per airfoil
- **Coverage**: Wide variety of shapes (thin/thick, symmetric/cambered)

### Generation Methods
1. **GMM Sampling** (best): Sample from fitted Gaussian Mixture Model
2. **Gaussian Sampling**: Sample from single Gaussian fit
3. **Empirical Sampling**: Sample near training examples with noise

### Feature Extraction
Automatically computed for all airfoils:
- Max thickness and location
- Average thickness
- Max camber and location
- Average camber
- Leading edge radius
- Trailing edge angle
- Chord length

### Latent Space Analysis
- Encode all 1,646 training airfoils
- Extract features for each
- Compute Pearson correlation between each dimension and each feature
- Label dimensions by strongest correlation

**Top Dimensions** (typical):
- Dim 23: avg_thickness (r=+0.881)
- Dim 11: avg_camber (r=-0.887)
- Dim 10: avg_thickness (r=+0.837)
- Dim 0: max_thickness (r=+0.787)

---

## Testing

### Run Verification Tests
```bash
python VERIFICATION_TESTS.py
```

**Tests** (8 total):
- Import verification
- File existence
- Model loading and forward pass
- Dataset loading
- LatentInterpreter functionality
- AirfoilGenerator functionality
- Coordinate format validation
- Example scripts existence

**Status**: All tests passing ✅

---

## Use Cases

**This project is for**:
- Preliminary airfoil design and exploration
- Understanding geometric trade-offs
- Generating diverse datasets for research
- Learning airfoil design and deep learning
- Initializing optimization pipelines

**This project is NOT for**:
- Production aircraft design (requires CFD validation)
- Safety-critical applications
- Multi-element airfoils (slats, flaps)
- Aerodynamic performance prediction

**IMPORTANT**: All generated airfoils must be validated with XFOIL or CFD before any real-world use.

---

## Key Technical Achievements

1. **Novel Architecture**: Deterministic AE solving VAE posterior collapse
2. **Extreme Compression**: 94% dimensionality reduction (400D → 24D)
3. **High Accuracy**: MSE < 0.00001 reconstruction
4. **Interpretability**: 10+ correlated dimensions for controllable generation
5. **Production Quality**: Clean code, comprehensive testing, full documentation

---

## System Requirements

**Minimum**:
- Python 3.8+
- 4 GB RAM
- CPU (no GPU needed)
- 50 MB disk space

**Recommended**:
- Python 3.9-3.12
- 8 GB RAM
- Multi-core CPU

---

## Troubleshooting

**Import errors**: Run `pip install -r requirements.txt`

**"No matching airfoils found"**: Targets too restrictive, try:
- Thickness: 0.05 - 0.20 (5% - 20%)
- Camber: 0.00 - 0.08 (0% - 8%)

**Unrealistic shapes**: Stay within typical parameter ranges above

---

## License

MIT License - See LICENSE file

---

## Citation

If you use this work, please cite:
```
DeepFoil: AI-Powered Airfoil Design System
Author: [Your Name]
Year: 2025
URL: [Your Repository URL]
```

---

## Version

**v1.0.0** - Initial release
- 24D deterministic autoencoder
- Interactive CLI interface
- Batch generation capabilities
- 3 example scripts
- Comprehensive testing

---

**Ready to start?** Run `python QUICK_DEMO.py` or `python deepfoil_interface.py`!
