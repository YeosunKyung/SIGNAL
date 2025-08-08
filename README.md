# SIGNAL Framework (Semantic Information-Guided Neuromorphic Action Learning)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14503792.svg)](https://doi.org/10.5281/zenodo.14503792)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Status: Research Archive](https://img.shields.io/badge/Status-Research%20Archive-orange)

## Overview

SIGNAL is a biologically-inspired framework for human action recognition that achieves **82.5% ± 1.2% accuracy** on the KTH dataset using neuromorphic visual processing principles.

## Key Features

- 🧠 **Biologically-inspired visual processing** including LGMD (Lobula Giant Movement Detector) modeling
- 📊 **540-dimensional feature extraction** with temporal aggregation
- 🔧 **Ensemble classification** combining SVM, Random Forest, and Gradient Boosting
- 📈 **Reproducible results** with 5-fold cross-validation

## Performance

| Metric | Value | Note |
|--------|-------|------|
| **Accuracy** | 82.5% ± 1.2% | Full KTH dataset, 5-fold CV |
| **Compression Ratio** | 14,400:1* | See [calculation details](COMPRESSION_RATIO_CALCULATION.md) |
| **Processing Time** | ~0.5s/video | On standard hardware |

*Compression ratio based on semantic information reduction. Alternative calculations yield 368:1 to 2,304:1 depending on assumptions.

### Per-Class Performance
- Walking: 90.0%
- Hand Waving: 88.0%
- Boxing: 87.0%
- Hand Clapping: 83.8%
- Jogging: 73.0%
- Running: 73.0%

## Installation

```bash
# Clone repository
git clone https://github.com/kyungpilpark/SIGNAL-neuromorphic-action-recognition.git
cd SIGNAL-neuromorphic-action-recognition

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from signal_model import SignalModel

# Initialize model
model = SignalModel()

# Train on KTH dataset
model.train(kth_path='path/to/KTH')

# Evaluate
accuracy = model.evaluate()
print(f"Accuracy: {accuracy:.1%}")
```

### Colab Notebooks
- `colab_signal_evaluation.py` - Full model evaluation
- `colab_signal_noise_robustness.py` - Noise robustness analysis
- `colab_lgmd_contribution_5fold.py` - Feature contribution analysis

## Known Limitations & Ongoing Research

### Current Limitations
1. **LGMD Contribution**: Current implementation shows minimal LGMD feature contribution. Investigation ongoing.
2. **Noise Robustness**: Performance degrades significantly below 20dB SNR
3. **Small Sample Performance**: Accuracy varies on small subsets

### Active Research Areas
- [ ] Improving LGMD neural modeling fidelity
- [ ] Enhancing noise robustness mechanisms
- [ ] Optimizing computational efficiency
- [ ] Extending to other action datasets

## Scientific Integrity Note

This repository represents ongoing research. While we achieve the reported performance metrics, we acknowledge certain biological features (particularly LGMD) do not contribute as theoretically expected. We maintain this code public for:

1. **Transparency** - All results are reproducible
2. **Education** - Learning from both successes and limitations
3. **Collaboration** - Inviting improvements from the community

## Repository Structure

```
.
├── notebooks/
│   ├── colab_signal_evaluation.py      # Main evaluation script
│   ├── colab_signal_features.py        # Feature extraction
│   └── colab_signal_noise_robustness.py # Robustness testing
├── figures/
│   └── signal_performance.png          # Performance visualizations
├── requirements.txt
└── README.md
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{signal2024,
  author = {Park, Kyungpil},
  title = {SIGNAL: Semantic Information-Guided Neuromorphic Action Learning},
  year = {2024},
  publisher = {GitHub},
  doi = {10.5281/zenodo.14503792},
  url = {https://github.com/kyungpilpark/SIGNAL-neuromorphic-action-recognition},
  note = {See repository README for known limitations}
}
```

## Future Improvements

We welcome contributions to address:
- Enhanced LGMD biological fidelity
- Improved noise robustness
- Extended dataset support
- Computational optimizations

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- KTH Action Dataset creators
- Biological vision research community
- Open science principles

---

**Note**: This is a research archive. For questions or collaborations, please open an issue or contact the authors.

**Development Transparency**: This repository's contribution history shows AI assistance (Claude) during the initial code implementation phase. All scientific concepts, research design, experimental decisions, and results are solely the work of the human authors. The AI tool was used only for coding assistance, similar to using an advanced IDE or code completion tool.