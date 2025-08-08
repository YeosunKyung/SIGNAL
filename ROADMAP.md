# SIGNAL Research Roadmap

## Current Status (December 2024)
- ✅ **82.5% accuracy** achieved on KTH dataset
- ⚠️ **LGMD contribution: 0%** - Critical issue to address
- 📊 **Ensemble model working** but biological components underperforming

## Phase 1: LGMD Investigation (Q1 2025)
### Goal: Understand why LGMD features show 0% contribution

1. **Variance Analysis**
   - [ ] Debug why LGMD features have zero variance
   - [ ] Test with synthetic looming stimuli
   - [ ] Compare with published LGMD implementations

2. **Parameter Tuning**
   - [ ] Grid search biological parameters (τ_m, τ_adapt)
   - [ ] Test different angular thresholds
   - [ ] Explore alternative LGMD formulations

3. **Feature Engineering**
   - [ ] Implement spike-based LGMD encoding
   - [ ] Test multi-scale LGMD responses
   - [ ] Add temporal derivatives

## Phase 2: Alternative Biological Models (Q2 2025)
### Goal: Explore other biological motion detectors

1. **Additional Neural Models**
   - [ ] Implement STMD (Small Target Motion Detector)
   - [ ] Add direction-selective cells (DS)
   - [ ] Explore ON/OFF pathway separation

2. **Hybrid Approaches**
   - [ ] Combine LGMD with attention mechanisms
   - [ ] Integrate with modern CNN features
   - [ ] Test neuromorphic preprocessing

## Phase 3: Performance Enhancement (Q3 2025)
### Goal: Achieve 85%+ accuracy while maintaining biological plausibility

1. **Architectural Improvements**
   - [ ] Implement spiking neural network backend
   - [ ] Add recurrent connections
   - [ ] Explore reservoir computing

2. **Robustness**
   - [ ] Improve noise resistance (target: 80%+ at 20dB)
   - [ ] Add motion blur handling
   - [ ] Test on additional datasets (UCF-101, HMDB-51)

## Phase 4: Real-world Applications (Q4 2025)
### Goal: Deploy in practical scenarios

1. **Efficiency Optimization**
   - [ ] Neuromorphic hardware implementation
   - [ ] Real-time processing (<10ms latency)
   - [ ] Mobile deployment

2. **Applications**
   - [ ] Drone collision avoidance
   - [ ] Autonomous vehicle safety
   - [ ] Robotics navigation

## Research Principles
1. **Transparency**: Document all failures and successes
2. **Reproducibility**: Provide runnable code for all experiments
3. **Biological Fidelity**: Balance performance with plausibility
4. **Open Science**: Share data, code, and insights

## Collaboration Opportunities
- Looking for collaborators with expertise in:
  - Neuromorphic engineering
  - Insect vision biology
  - Spiking neural networks
  - Computer vision

## Metrics for Success
- [ ] LGMD contribution > 10%
- [ ] Overall accuracy > 85%
- [ ] Noise robustness > 80% at 20dB
- [ ] Processing speed < 10ms/frame
- [ ] Successful deployment in one real-world application

## Contact
For collaboration or questions:
- GitHub Issues: [Project Repository](https://github.com/YeosunKyung/SIGNAL)
- Email: [your-email]

---
*Last updated: December 2024*