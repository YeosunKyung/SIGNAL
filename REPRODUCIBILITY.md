# Reproducibility Checklist for SIGNAL

This document ensures reproducibility of results reported in the IEEE ACCESS paper.

## ✅ Environment Setup

- [ ] Python 3.7 or higher installed
- [ ] All dependencies from `requirements.txt` installed
- [ ] KTH dataset downloaded and placed in correct directory

## ✅ Pre-flight Checks

Run these commands to verify your setup:

```bash
# 1. Verify installation
python verify_installation.py

# 2. Check Python version
python --version  # Should be 3.7+

# 3. Test imports
python -c "from SIGNAL_model import SIGNALModel; print('✅ Import successful')"
```

## ✅ Reproducing Paper Results

### Step 1: Download KTH Dataset
```bash
# Download from: https://www.csc.kth.se/cvap/actions/
# Extract to: ./KTH_dataset/
# Structure should be:
# KTH_dataset/
#   ├── boxing/
#   ├── handclapping/
#   ├── handwaving/
#   ├── jogging/
#   ├── running/
#   └── walking/
```

### Step 2: Run Main Experiment
```bash
# Set random seed for reproducibility
export PYTHONHASHSEED=0

# Run the model
python SIGNAL_model.py
```

### Step 3: Expected Results
- **Accuracy**: 80.8% ± 2%
- **Compression Ratio**: 14,400:1
- **Selected Features**: 100
- **Training Time**: ~10-15 minutes (CPU)

## ✅ Troubleshooting

### Common Issues:

1. **OpenCV Import Error**
   ```bash
   pip install opencv-python --upgrade
   ```

2. **Memory Error with Large Dataset**
   - Reduce `max_videos_per_class` in the code
   - Use subset of data for testing

3. **Different Results**
   - Ensure random seed is set: `PYTHONHASHSEED=0`
   - Use exact versions from `requirements.txt`
   - Verify dataset is complete

## ✅ Hardware Requirements

- **Minimum RAM**: 8GB
- **Recommended RAM**: 16GB
- **Storage**: 2GB for dataset + code
- **GPU**: Not required (CPU-only implementation)

## ✅ Validation Points

The following should be verified:

1. **Feature Extraction**
   - 90 features per frame pair (45 × 2 scales)
   - 540 total features after temporal aggregation
   - 100 features after selection

2. **Biological Parameters**
   - τ_m = 8ms
   - τ_adapt = 132ms
   - Angle threshold = 17.5°

3. **Model Architecture**
   - Ensemble: SVM + Random Forest + Gradient Boosting
   - Weighted voting based on training performance

## 📧 Support

If you encounter issues reproducing results:
- Email: yeosun.kyung@yonsei.ac.kr
- GitHub Issues: https://github.com/YeosunKyung/SIGNAL/issues

## 📊 Computational Reproducibility

This code has been tested on:
- macOS 11.x, 12.x
- Ubuntu 18.04, 20.04, 22.04
- Windows 10, 11 (with WSL)
- Google Colab (Free tier)