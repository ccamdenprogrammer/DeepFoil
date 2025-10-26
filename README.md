# DeepFoil: Deep Learning for Airfoil Design

<div align="center">

**A Deterministic Autoencoder for Generative Airfoil Design**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Results](#results) • [Timeline](TIMELINE.md)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Training](#training)
- [Usage Examples](#usage-examples)
- [Results](#results)
- [Roadmap](#roadmap)
- [Lessons Learned](#lessons-learned)
- [License](#license)

---

## Overview

DeepFoil is a deep learning system for automated airfoil design using a **deterministic Autoencoder**. After extensive experimentation with Variational Autoencoders (VAEs), we discovered that a deterministic approach provides superior reconstruction quality and diversity without posterior collapse issues.

The system learns a compact, continuous latent representation of airfoil geometries from the UIUC Airfoil Database and enables:

- ✨ **Generative Design**: Create novel, aerodynamically-valid airfoil shapes
- 🎯 **High-Fidelity Reconstruction**: Reconstruct airfoils with 0.000004 MSE
- 🔄 **Latent Space Exploration**: Smoothly interpolate between existing designs
- 🚀 **Fast Generation**: Generate new designs in milliseconds

### Why Autoencoder vs VAE?

During development, we discovered that VAEs suffer from **posterior collapse** in this domain - the latent space collapses and all reconstructions become identical. After extensive debugging (documented in `diagnose_collapse.py`), we switched to a deterministic Autoencoder that:

1. **Eliminates Collapse**: No KL divergence penalty means no collapse
2. **Better Reconstruction**: 5.5× lower MSE than best VAE attempt
3. **Maintained Diversity**: Latent codes remain diverse (std = 0.155)
4. **Simpler Training**: Fewer hyperparameters to tune

Generation is achieved by fitting a Gaussian distribution to the learned latent space, providing similar sampling capabilities to a VAE without the training difficulties.

### How It Works

```
Input Airfoil (200 points) → Encoder → Latent Vector (24D) → Decoder → Reconstructed Airfoil
                                           ↓
                                 Fit Gaussian Distribution
                                           ↓
                                 Sample → Generate Novel Airfoils
```

## Features

### Core Capabilities

- ✅ **Exceptional Reconstruction**: MSE of 0.000004 (5.5× better than VAE)
- ✅ **Perfect Smoothness**: Mean curvature of 0.000348 (3.2× smoother than VAE)
- ✅ **100% TE Closure**: All generated airfoils have properly closed trailing edges
- ✅ **Diverse Generation**: Latent space maintains high diversity (std = 0.155)
- ✅ **Fast Training**: 300 epochs in ~2 hours on CPU
- ✅ **No Posterior Collapse**: Deterministic approach eliminates VAE collapse issues

### Technical Features

- 🔧 **24D Latent Space**: Optimal balance between expressiveness and efficiency
- 🔧 **Progressive Smoothness**: Gradually increase smoothness constraint during training
- 🔧 **AdamW Optimizer**: With weight decay for better generalization
- 🔧 **Adaptive Learning Rate**: ReduceLROnPlateau scheduler
- 🔧 **Minimal Dropout**: 2% dropout for slight regularization
- 🔧 **Gaussian Generation**: Simple sampling from learned latent distribution

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Reconstruction MSE** | 0.000004 | Mean squared error (5.5× better than VAE) |
| **Smoothness (Curvature)** | 0.000348 | Second derivative of y-coordinates (3.2× better) |
| **Trailing Edge Closure** | 0.003422 | Mean TE distance (100% < 0.05) |
| **Latent Diversity** | 0.155 | Standard deviation across latent dimensions |
| **Training Time** | ~2 hours | 300 epochs on CPU |
| **Inference Speed** | <1ms | Single airfoil generation |

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/ccamdenprogrammer/DeepFoil.git
cd DeepFoil

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib scipy
```

---

## Quick Start

### 1. Download and Prepare Data

```bash
# Download UIUC Airfoil Database
python src/data/download_uiuc.py

# Create processed dataset
python src/data/create_dataset.py
```

### 2. Train the Model

```bash
# Train autoencoder (300 epochs, ~2 hours)
python src/training/train_ae.py
```

Training progress will be displayed every 10 epochs:
```
Epoch  10/300 | Train: 0.000595 | Val: 0.000284 | Recon: 0.000592 | Smooth: 0.000026
Epoch  20/300 | Train: 0.000549 | Val: 0.000303 | Recon: 0.000542 | Smooth: 0.000024
...
```

### 3. Visualize Results

```bash
# Generate visualizations and analysis
python visualize_ae.py
```

This creates:
- `outputs/plots/ae_reconstructions.png` - Original vs reconstructed airfoils
- `outputs/plots/ae_generated_airfoils.png` - 10 generated novel airfoils
- `outputs/plots/ae_training_history.png` - Training curves

---

## Project Structure

```
DeepFoil/
├── src/
│   ├── data/
│   │   ├── download_uiuc.py      # Download UIUC database
│   │   └── create_dataset.py     # Process into PyTorch dataset
│   ├── models/
│   │   ├── airfoil_ae.py         # Autoencoder architecture
│   │   └── airfoil_vae.py        # VAE (legacy, has collapse issues)
│   └── training/
│       ├── train_ae.py           # Autoencoder training script
│       └── train_vae.py          # VAE training (for comparison)
├── visualize_ae.py               # Visualization and analysis
├── analyze_model.py              # Quantitative analysis tool
├── diagnose_collapse.py          # VAE collapse diagnostic
├── detailed_comparison.py        # Detailed reconstruction comparison
├── README.md
└── TIMELINE.md                   # Complete 12-week development timeline
```

---

## Architecture

### Autoencoder Design

```python
Input: 400 values (200 x,y coordinate pairs)

Encoder:
  400 → Linear(256) → LayerNorm → SiLU → Dropout(0.02)
  256 → Linear(128) → LayerNorm → SiLU → Dropout(0.02)
  128 → Linear(64)  → LayerNorm → SiLU → Dropout(0.02)
  64  → Linear(24)  # Latent space

Decoder:
  24  → Linear(64)  → SiLU
  64  → Linear(128) → SiLU
  128 → Linear(256) → SiLU
  256 → Linear(400)

Output: 400 values (reconstructed airfoil)
```

**Key Architectural Decisions:**

1. **24D Latent Space**: Provides sufficient expressiveness while remaining interpretable
2. **LayerNorm in Encoder**: Stabilizes training, especially important for varying input scales
3. **No Normalization in Decoder**: Allows decoder full flexibility to reconstruct details
4. **Minimal Dropout (2%)**: Just enough regularization without harming reconstruction
5. **SiLU Activation**: Smooth, non-monotonic activation performs better than ReLU

### Loss Function

```python
total_loss = recon_loss + smoothness_weight * smooth_loss

where:
  recon_loss = MSE(reconstructed, original)
  smooth_loss = mean(d²y/dx²)²  # Second derivative penalty
  smoothness_weight: 0.0 → 2.0 (progressive over 150 epochs)
```

**Why This Loss?**

- **Reconstruction (MSE)**: Ensures accurate geometry reproduction
- **Smoothness Penalty**: Enforces aerodynamically-valid smooth curves
- **Progressive Weighting**: Allows model to learn reconstruction first, then refine smoothness

---

## Training

### Optimized Hyperparameters

```python
LATENT_DIM = 24
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 300
WEIGHT_DECAY = 1e-5

SMOOTHNESS_START = 0.0
SMOOTHNESS_END = 2.0
SMOOTHNESS_RAMP_EPOCHS = 150
```

### Training Strategy

1. **Phase 1 (Epochs 1-150)**: Progressive smoothness
   - Start with pure reconstruction (smoothness = 0)
   - Gradually increase to smoothness = 2.0
   - Allows model to learn basic shapes first

2. **Phase 2 (Epochs 151-300)**: Refinement
   - Fixed smoothness = 2.0
   - Model refines details and improves quality
   - Learning rate reduced by scheduler as needed

### Training Tips

- **Monitor Smoothness**: Should decrease steadily
- **Watch Validation Loss**: Should track training loss closely
- **Check Reconstructions**: Visualize every 50 epochs
- **Learning Rate**: Will automatically reduce on plateau

---

## Usage Examples

### Generate New Airfoils

```python
import torch
import numpy as np
from src.models.airfoil_ae import AirfoilAE

# Load trained model
device = torch.device("cpu")
checkpoint = torch.load("models/airfoil_ae/best_model.pth", map_location=device)
model = AirfoilAE(input_dim=400, latent_dim=24)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load latent distribution (computed from dataset)
latent_mean = np.load("latent_mean.npy")  # From visualize_ae.py
latent_cov = np.load("latent_cov.npy")

# Generate 10 new airfoils
with torch.no_grad():
    for i in range(10):
        # Sample from learned distribution
        z = np.random.multivariate_normal(latent_mean, latent_cov, size=1)
        z_tensor = torch.tensor(z, dtype=torch.float32)

        # Decode to airfoil
        airfoil = model.decode(z_tensor)
        coords = airfoil.numpy().reshape(-1, 2)

        # Plot
        plt.plot(coords[:, 0], coords[:, 1])
        plt.axis('equal')
        plt.show()
```

### Reconstruct Existing Airfoil

```python
from src.data.create_dataset import AirfoilDataset

# Load dataset
dataset = AirfoilDataset.load("data/processed/airfoil_dataset.pkl")

# Encode and reconstruct
original = dataset[0].unsqueeze(0)
reconstructed, latent = model(original)

# Compare
orig_coords = original.numpy().reshape(-1, 2)
recon_coords = reconstructed.numpy().reshape(-1, 2)

plt.plot(orig_coords[:, 0], orig_coords[:, 1], 'b-', label='Original')
plt.plot(recon_coords[:, 0], recon_coords[:, 1], 'r--', label='Reconstructed')
plt.legend()
plt.show()
```

### Interpolate Between Airfoils

```python
# Encode two airfoils
airfoil1 = dataset[10].unsqueeze(0)
airfoil2 = dataset[50].unsqueeze(0)

z1 = model.encode(airfoil1)
z2 = model.encode(airfoil2)

# Interpolate
for alpha in np.linspace(0, 1, 11):
    z_interp = (1 - alpha) * z1 + alpha * z2
    airfoil_interp = model.decode(z_interp)
    coords = airfoil_interp.detach().numpy().reshape(-1, 2)
    plt.plot(coords[:, 0], coords[:, 1], alpha=0.3 + 0.7*alpha)

plt.axis('equal')
plt.show()
```

---

## Results

### Reconstructions

The model achieves near-perfect reconstruction of airfoils from the dataset:

| Sample | Original Shape | Reconstruction MSE | Visual Quality |
|--------|---------------|-------------------|----------------|
| 1 | Thin symmetric | 0.000001 | Perfect |
| 2 | Thick cambered | 0.000002 | Perfect |
| 3 | Medium airfoil | 0.000001 | Perfect |
| 4 | Cambered | 0.000013 | Excellent |
| 5 | Symmetric | 0.000001 | Perfect |
| 6 | Diverse | 0.000002 | Perfect |

**Key Observations:**
- Reconstructions are visually indistinguishable from originals
- All geometric features preserved (camber, thickness, curvature)
- No averaging or smoothing artifacts

### Generated Airfoils

Generated airfoils exhibit excellent quality:

- ✅ **Smooth Curves**: Mean curvature 0.000348 (extremely smooth)
- ✅ **Closed TE**: 100% have trailing edge gap < 0.005
- ✅ **Diverse**: Wide variety of shapes (thin, thick, symmetric, cambered)
- ✅ **Realistic**: All geometries are aerodynamically plausible

### Comparison: VAE vs Autoencoder

| Metric | VAE (Best Attempt) | Autoencoder | Improvement |
|--------|-------------------|-------------|-------------|
| Reconstruction MSE | 0.000022 | **0.000004** | **5.5×** |
| Smoothness | 0.001125 | **0.000348** | **3.2×** |
| TE Closure | 0.004197 | **0.003422** | **1.2×** |
| Latent Diversity | 0.188 | **0.155** | Maintained |
| Posterior Collapse | ❌ Yes | ✅ No | Solved |

---

## Roadmap

### ✅ Completed (Weeks 1-4)

- [x] UIUC database download and processing
- [x] Deterministic Autoencoder architecture
- [x] Optimized training pipeline
- [x] High-quality generation system
- [x] Comprehensive visualization tools
- [x] Posterior collapse diagnosis and fix

### 🚧 In Progress (Weeks 5-7)

- [ ] XFOIL integration for aerodynamic analysis
- [ ] Performance dataset generation (Cl, Cd, L/D)
- [ ] Batch analysis scripts

### 🔜 Planned (Weeks 8-12)

- [ ] Performance predictor neural network
- [ ] Optimization system (generate from requirements)
- [ ] Web interface for airfoil generation
- [ ] Comprehensive testing and validation
- [ ] Documentation and presentation

For detailed week-by-week breakdown, see **[TIMELINE.md](TIMELINE.md)**

---

## Lessons Learned

### VAE Posterior Collapse

We spent significant effort trying to make VAEs work but encountered severe **posterior collapse** where all reconstructions became identical. Key insights:

1. **High Smoothness Weight → Collapse**: Smoothness penalty dominated, decoder ignored latent codes
2. **Free Bits Mask Problem**: Free bits maintain KL artificially but don't prevent collapse
3. **Beta-Annealing Insufficient**: Even with careful beta scheduling, collapse persisted
4. **Variance Penalty Backfires**: Adding variance penalties actually worsened collapse

**Solution**: Switch to deterministic Autoencoder + Gaussian sampling

### Training Insights

1. **Progressive Smoothness is Critical**: Starting with smoothness=0 and gradually increasing to 2.0 allows model to learn reconstruction first
2. **Small Batches Help**: Batch size of 32 provides better gradient estimates than 64
3. **Minimal Dropout Works Best**: 2% dropout provides slight regularization without harming quality
4. **AdamW > Adam**: Weight decay improves generalization
5. **300 Epochs Optimal**: Further training shows diminishing returns

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{deepfoil2024,
  author = {Camden Crace},
  title = {DeepFoil: Deep Learning for Airfoil Design},
  year = {2024},
  url = {https://github.com/ccamdenprogrammer/DeepFoil}
}
```

---

## Acknowledgments

- **UIUC Airfoil Database**: For providing the comprehensive airfoil dataset
- **PyTorch**: For the excellent deep learning framework
- **Posterior Collapse Debugging**: Extensive testing revealed fundamental VAE limitations in this domain

---

<div align="center">

**Ready to generate airfoils?** Start with [Quick Start](#quick-start)

For the complete development timeline, see **[TIMELINE.md](TIMELINE.md)**

</div>
