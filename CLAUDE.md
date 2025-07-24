# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a biologically-inspired action recognition system using LGMD (Lobula Giant Movement Detector) neural mechanisms combined with hyperbolic contrastive learning. The project implements a motion detection and classification pipeline inspired by insect visual systems.

## Build and Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main pipeline
python improved_lgmd_hyperbolic_pipeline.py

# Run specific components
python part1_lgmd_encoder.py          # LGMD feature extraction only
python part2_hyperbolic_learning.py   # Hyperbolic learning component
python part3_semantic_comm_analysis.py # Semantic communication analysis

# Quick start for Google Colab
python quick_start_colab.py

# Run comprehensive analysis
python comprehensive_analysis_pipeline.py
```

## Architecture Overview

The system consists of four main components connected in a pipeline:

1. **LGMD Encoder** (`ImprovedLGMDEncoder` class)
   - Converts video frames to spike trains using motion detection
   - Uses leaky integration to produce analog voltage outputs
   - Implements directional selectivity and feed-forward inhibition
   - Key parameters configured in `config.py`

2. **Hyperbolic Embedding** 
   - Projects features to Poincaré ball manifold for hierarchical representations
   - Uses contrastive learning with positive/negative pairs
   - Implemented using geoopt library

3. **Structural Plasticity**
   - Prototype selection based on novelty and redundancy
   - Adaptive thresholds with fallback mechanisms
   - Ensures at least one prototype per class

4. **Semantic Decoder**
   - Multi-layer perceptron for final classification
   - Configurable architecture with dropout regularization

## Key Configuration

Primary configuration is in `config.py`:
- `LGMD_CONFIG`: Core encoder parameters (patch_size, leak_rate, threshold)
- `DATASET_CONFIG`: Dataset specifications
- `FEATURE_CONFIG`: Feature enhancement settings
- `TRAINING_CONFIG`: Cross-validation and training parameters

## Testing Approach

The project uses cross-validation evaluation rather than unit tests:
- `robust_evaluation()` function performs k-fold cross-validation
- Statistical significance testing via paired t-tests
- Ablation studies to validate component contributions

No pytest/unittest framework is used - validation is done through evaluation pipelines.

## Important Design Decisions

1. **Analog vs Binary Output**: The system generates binary spikes but outputs analog voltages through leaky integration for richer information content.

2. **Hyperbolic Space**: Uses Poincaré ball embeddings to capture hierarchical relationships in action classes.

3. **Prototype Selection**: Dynamic prototype selection with novelty/redundancy thresholds to balance representation and efficiency.

4. **Multiple Entry Points**: Different scripts for various use cases (quick_start for Colab, comprehensive_analysis for full evaluation, etc.)