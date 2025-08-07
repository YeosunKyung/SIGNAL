# SIGNAL: Semantic Information Guided Neuromorphic Action Learning

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/signal-semantic-information-guided/action-recognition-on-kth)](https://paperswithcode.com/sota/action-recognition-on-kth)

A biologically-inspired approach to video action recognition achieving extreme compression ratios while maintaining high accuracy.

## 🚀 Key Features

- **80.8% accuracy** on KTH action recognition dataset
- **14,400:1 compression ratio** - from 46MB/s to 3.2KB/s
- **100 biologically-inspired features** selected from 540 candidates
- **Real-time processing** capability
- **LGMD-based** collision detection mechanism from locust visual system

## 📊 Performance

| Metric | Value |
|--------|-------|
| Accuracy | 80.8% |
| Compression Ratio | 14,400:1 |
| Features | 100 |
| Processing Speed | ~30 fps |
| Model Size | < 10MB |

## 🧬 Biological Inspiration

SIGNAL is inspired by the Lobula Giant Movement Detector (LGMD) neuron found in locusts, which responds to looming stimuli and collision threats. Key biological parameters:

- Membrane time constant (τ_m): 8ms
- Adaptation time constant (τ_adapt): 132ms
- Preferred angle threshold: 17.5°

## 📁 Repository Structure

```
SIGNAL/
├── SIGNAL_model.py          # Main model implementation
├── requirements.txt         # Python dependencies
├── demo.ipynb              # Interactive demo notebook
├── data/
│   └── KTH_dataset/        # Place KTH dataset here
├── results/
│   ├── figures/            # Generated figures
│   └── models/             # Saved models
└── docs/
    └── paper.pdf           # IEEE ACCESS paper
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yeosunkyung/SIGNAL.git
cd SIGNAL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the KTH dataset:
```bash
# Download from: https://www.csc.kth.se/cvap/actions/
# Extract to data/KTH_dataset/
```

## 🚀 Quick Start

```python
from SIGNAL_model import SIGNALModel

# Initialize model
model = SIGNALModel()

# Load dataset
X, y, actions = model.load_dataset("data/KTH_dataset")

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model.train(X_train, y_train)

# Evaluate
results = model.evaluate(X_test, y_test, actions)
print(f"Accuracy: {results['accuracy']:.1%}")
```

## 📈 Results

### Confusion Matrix
![Confusion Matrix](results/figures/confusion_matrix.png)

### Feature Importance
![Feature Analysis](results/figures/feature_analysis.png)

### Compression Comparison
| Method | Compression Ratio | Accuracy |
|--------|------------------|----------|
| Raw Video | 1:1 | 100% |
| JPEG | 10:1 | 95% |
| H.264 | 100:1 | 90% |
| H.265 | 200:1 | 88% |
| **SIGNAL** | **14,400:1** | **80.8%** |

## 🔬 Technical Details

### Feature Extraction Pipeline

1. **Multi-scale Processing**: Two scales (96×96, 64×64)
2. **Motion Features**: 7 features per scale
3. **Optical Flow**: 8 features per scale
4. **Expansion Detection**: 8 features per scale
5. **LGMD Response**: 5 features per scale
6. **Directional Histogram**: 8 bins
7. **Spatial Grid**: 3×3 regions

Total: 45 features × 2 scales = 90 features per frame pair

### Temporal Aggregation

- Mean, Standard Deviation, Maximum, 75th Percentile
- Early-to-Middle and Middle-to-Late temporal dynamics

Total: 90 × 6 = 540 features → 100 selected features

### Model Architecture

Weighted ensemble of:
- Support Vector Machine (SVM) with RBF kernel
- Random Forest with 300 trees
- Gradient Boosting with 200 estimators

## 📝 Citation

If you use SIGNAL in your research, please cite:

```bibtex
@article{signal2024,
  title={SIGNAL: Semantic Information Guided Neuromorphic Action Learning},
  author={Kyung, Yeosun and Kim, Seong-Lyun},
  journal={IEEE ACCESS},
  year={2025},
  doi={/ACCESS.2025.XXXXXXX}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- KTH Royal Institute of Technology for the action dataset
- Neuroscience research on LGMD neurons
- IEEE ACCESS for publication

## 🔬 Reproducibility

This repository is committed to reproducible research:

- **Environment**: Tested on Python 3.7-3.10
- **Dataset**: Uses public KTH Action Dataset
- **Random Seeds**: Fixed at 42 for all experiments
- **Hardware**: CPU-only implementation (no GPU required)

### Running Experiments

```bash
# Reproduce paper results
python SIGNAL_model.py

# With specific random seed
PYTHONHASHSEED=0 python SIGNAL_model.py
```

### Citing this Work

For citations, please use the DOI from Zenodo or cite our paper:

```
@software{signal2025,
  author = {Kyung, Yeosun},
  title = {SIGNAL: Semantic Information Guided Neuromorphic Action Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YeosunKyung/SIGNAL},
  doi = {10.5281/zenodo.XXXXXX}
}
```

## 📧 Contact

- Author: Yeosun Kyung
- Email: yeosun.kyung@yonsei.ac.kr
- Lab: [Lab Website]
- Paper: [IEEE ACCESS Link]

---

**Note**: This is a research implementation. For production use, additional optimizations may be needed.