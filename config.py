"""
Configuration file for LGMD Encoder
Contains all hyperparameters and settings for the LGMD feature extraction
"""

# LGMD Encoder Parameters
LGMD_CONFIG = {
    'patch_size': 8,
    'leak_rate': 0.95,
    'threshold': 0.1,
    'feedforward_inhibition': 0.1,
    'directional_weight': 0.3
}

# Dataset Parameters
DATASET_CONFIG = {
    'frame_height': 120,
    'frame_width': 160,
    'num_frames': 20,
    'synthetic_videos': 600
}

# Feature Enhancement Parameters
FEATURE_CONFIG = {
    'variance_threshold_percentile': 2,  # Keep top 98% variance features
    'max_features': 150,  # Maximum number of features to keep
    'interaction_spacing': 3,  # Spacing between interaction features
    'gaussian_sigma': 0.7  # Sigma for Gaussian smoothing
}

# Training Parameters
TRAINING_CONFIG = {
    'cv_folds': 3,
    'random_state': 42,
    'test_size': 0.2,
    'n_jobs': -1  # Use all CPU cores
}

# Google Drive Paths (Colab-specific)
PATHS = {
    'drive_path': "/content/drive/MyDrive/KTH_dataset",
    'features_save_path': "/content/drive/MyDrive/lgmd_features.npz",
    'results_save_path': "/content/drive/MyDrive/lgmd_results.json"
}

# Debug Settings
DEBUG_CONFIG = {
    'verbose': True,
    'debug_videos': 3,  # Number of videos to debug
    'progress_interval': 10  # Progress reporting interval (%)
}

# Performance Settings
PERFORMANCE_CONFIG = {
    'use_gpu': True,
    'batch_size': 32,
    'memory_efficient': True,
    'clear_gpu_cache': True
} 