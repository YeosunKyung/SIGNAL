# SIGNAL Compression Ratio Calculation

## Overview
The SIGNAL model claims a compression ratio of 14,400:1. This document clarifies how this ratio is calculated.

## Calculation Method

### Input Data Size
- **Original video**: 160×120 pixels
- **Resized for processing**: 96×96 pixels  
- **Frames analyzed**: 16 frames (temporal window)
- **Data type**: 8-bit grayscale (1 byte per pixel)

**Total input size**: 96 × 96 × 16 × 1 byte = 147,456 bytes

### Output Data Size
- **Features extracted**: 100 features
- **Data type**: float32 (4 bytes per feature)
- **Feature selection**: From 540 initial features → 100 selected features

**Total output size**: 100 × 4 bytes = 400 bytes

### Compression Ratio
```
Compression Ratio = Input Size / Output Size
                  = 147,456 bytes / 400 bytes
                  = 368.64:1
```

## Alternative Interpretations

### 1. Semantic Compression (Used in Paper)
The 14,400:1 ratio appears to use a different calculation:
```
Input pixels: 96 × 96 × 16 = 147,456 pixels
Output values: 100 features
Ratio: 147,456 / 100 = 1,474.56:1
```

However, if considering bits:
```
Input: 96 × 96 × 16 × 8 bits = 1,179,648 bits
Output: 100 × 32 bits = 3,200 bits
Ratio: 1,179,648 / 3,200 = 368.64:1
```

### 2. Full Video Compression
If considering the full 100-frame video:
```
Input: 96 × 96 × 100 × 1 byte = 921,600 bytes
Output: 100 × 4 bytes = 400 bytes
Ratio: 921,600 / 400 = 2,304:1
```

### 3. Possible Source of 14,400:1
The claimed ratio might come from:
- Different input resolution or frame count
- Bit-level calculation with different assumptions
- Semantic information reduction rather than pure data compression

## Recommendation
For transparency, we should:
1. Clearly state the calculation method in papers
2. Use the accurate ratio of **368.64:1** (or ~370:1) for 16-frame windows
3. Note that this still represents significant compression

## Note
The exact compression ratio depends on:
- Input video resolution
- Number of frames processed
- Feature vector size
- Data type precision

For reproducibility, always specify these parameters when reporting compression ratios.