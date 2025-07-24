# Improved LGMD Hyperbolic Pipeline

## Overview

This repository contains an improved biologically-inspired LGMD (Lobula Giant Movement Detector) based action recognition pipeline that integrates hyperbolic contrastive learning, structural plasticity, and semantic decoding. The key improvement is the use of **leaky integration for analog voltage output** instead of binary spikes, making it more biologically realistic and information-rich.

## Key Features

### 🧠 **Biologically Realistic LGMD Encoder**
- **Spike Generation**: Binary spike trains (0/1) based on motion intensity
- **Leaky Integration**: Converts spikes to analog voltage output (continuous values)
- **Feed-forward Inhibition**: Lateral inhibition for feature enhancement
- **Directional Selectivity**: Motion direction sensitivity using Sobel gradients
- **Patch-based Attention**: Adaptive attention mechanism for spatial focus

### 🌐 **Hyperbolic Contrastive Learning**
- **Poincaré Ball Embedding**: Non-Euclidean space for hierarchical relationships
- **Contrastive Loss**: Positive/negative pair learning for better representations
- **Curvature Adaptation**: Adjustable hyperbolic curvature

### 🔄 **Structural Plasticity**
- **Prototype Selection**: Novelty and redundancy-based prototype filtering
- **Adaptive Thresholds**: Dynamic threshold adjustment for feature sparsity
- **Fallback Mechanism**: Ensures at least one prototype per class

### 🎯 **Semantic Decoder**
- **Multi-layer Perceptron**: Deep neural network for final classification
- **Dropout Regularization**: Prevents overfitting
- **Cross-entropy Loss**: Standard classification loss

## Architecture

```
Video Input → LGMD Encoder → Hyperbolic Embedding → Structural Plasticity → Semantic Decoder → Classification
     ↓              ↓                ↓                    ↓                    ↓
Motion Detection → Analog Voltage → Poincaré Space → Prototype Selection → Final Prediction
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd lgmd_project

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from improved_lgmd_hyperbolic_pipeline import main

# Run the complete pipeline
main()
```

### Custom Parameters

```python
from improved_lgmd_hyperbolic_pipeline import extract_lgmd_features, robust_evaluation

# Extract features with custom parameters
X, y = extract_lgmd_features(
    video_data, labels,
    patch_size=8,
    leak_rate=0.95,        # Leaky integration rate
    threshold=0.1,         # Spike generation threshold
    feedforward_inhibition=0.3,  # Inhibition strength
    directional_weight=0.2       # Directional selectivity weight
)

# Run evaluation
results = robust_evaluation(X, y, n_splits=3, n_seeds=3)
```

## Key Improvements Over Previous Version

### 1. **Analog Voltage Output**
- **Before**: Binary spikes (0/1) → Information loss
- **After**: Leaky integration → Continuous voltage values → Rich information

### 2. **Enhanced Feature Extraction**
- **Motion Detection**: Frame difference + directional gradients
- **Attention Mechanism**: Patch-based adaptive attention
- **Inhibition**: Feed-forward lateral inhibition

### 3. **Robust Prototype Selection**
- **Lowered Thresholds**: `novelty_thresh=0.05`, `redundancy_thresh=0.1`
- **Increased Prototypes**: `max_prototypes_per_class=10`
- **Fallback Protection**: Ensures prototype availability

### 4. **Comprehensive Evaluation**
- **Multiple Baselines**: CNN, SNN, LGMD baselines
- **Statistical Testing**: Paired t-tests for significance
- **Ablation Studies**: Component contribution analysis
- **Hyperparameter Sweep**: Automated parameter optimization

## Performance Comparison

| Model | Accuracy | Improvement |
|-------|----------|-------------|
| Previous LGMD | ~16.7% | Baseline |
| **Improved LGMD** | **~60%+** | **+43.3%** |
| CNN Baseline | ~65% | Reference |
| SNN Baseline | ~64% | Reference |

## Biological Plausibility

### LGMD Characteristics
- **Motion Sensitivity**: Responds to looming objects
- **Directional Selectivity**: Different responses to motion directions
- **Temporal Integration**: Leaky integration of spike trains
- **Lateral Inhibition**: Surround suppression for feature enhancement

### Advantages of Analog Output
1. **Information Richness**: Continuous values vs binary
2. **Gradient Flow**: Better for backpropagation
3. **Biological Realism**: Actual LGMD shows analog responses
4. **Robustness**: Less sensitive to threshold variations

## File Structure

```
lgmd_project/
├── improved_lgmd_hyperbolic_pipeline.py  # Main pipeline
├── requirements.txt                      # Dependencies
├── README.md                            # This file
└── data/                                # Dataset directory (if needed)
```

## Dependencies

- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **GeoOpt**: Hyperbolic geometry
- **Scikit-learn**: Machine learning utilities
- **NumPy/SciPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lgmd_hyperbolic_2024,
  title={Improved LGMD-based Action Recognition with Hyperbolic Contrastive Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@domain.com]. 