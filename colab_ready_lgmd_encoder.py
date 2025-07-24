# LGMD Encoder for Google Colab
# Copy and paste this entire code into a Colab cell and run it

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import warnings
import os
from collections import Counter
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import geoopt
from geoopt.manifolds import PoincareBall
from google.colab import drive
import zipfile
import glob
from scipy.ndimage import gaussian_filter, convolve
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Mount Google Drive (for Colab)
def mount_drive():
    """Mount Google Drive for Colab"""
    try:
        drive.mount('/content/drive')
        print("Google Drive mounted successfully!")
        return True
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        return False

class ImprovedLGMDEncoder:
    """
    생물학적으로 현실적인 LGMD encoder
    - 스파이크는 바이너리로 생성하되
    - 출력은 leaky integration된 아날로그 전압값
    - GPU 최적화된 convolution 연산 사용
    """
    
    def __init__(self, patch_size=8, leak_rate=0.95, threshold=0.1, 
                 feedforward_inhibition=0.3, directional_weight=0.2):
        self.patch_size = patch_size
        self.leak_rate = leak_rate
        self.threshold = threshold
        self.feedforward_inhibition = feedforward_inhibition
        self.directional_weight = directional_weight
        
    def extract_motion_features(self, frames):
        """Motion intensity extraction with directional selectivity (GPU optimized)"""
        if len(frames) < 2:
            return np.zeros((frames[0].shape[0] // self.patch_size, 
                           frames[0].shape[1] // self.patch_size))
        
        motion_features = []
        for i in range(1, len(frames)):
            # Frame difference
            diff = cv2.absdiff(frames[i], frames[i-1])
            
            # Ensure diff is not all zeros
            if np.sum(diff) == 0:
                # Add small noise to create motion
                diff = np.random.rand(*diff.shape) * 0.01
            
            # GPU-optimized directional derivatives using PyTorch
            diff_tensor = torch.from_numpy(diff).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Sobel kernels for directional detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # GPU convolution for directional derivatives
            grad_x = F.conv2d(diff_tensor, sobel_x, padding=1)
            grad_y = F.conv2d(diff_tensor, sobel_y, padding=1)
            
            # Directional motion intensity
            directional_motion = torch.sqrt(grad_x**2 + grad_y**2)
            
            # Convert back to numpy
            directional_motion = directional_motion.squeeze().cpu().numpy()
            diff = diff_tensor.squeeze().cpu().numpy()
            
            # Combine with regular motion
            combined_motion = (1 - self.directional_weight) * diff + \
                            self.directional_weight * directional_motion
            
            motion_features.append(combined_motion)
        
        motion_intensity = np.mean(motion_features, axis=0)
        
        # Debug: Check motion intensity (only for first few videos)
        global motion_debug_counter
        if 'motion_debug_counter' not in globals():
            motion_debug_counter = 0
        motion_debug_counter += 1
        
        if motion_debug_counter <= 3:  # Only print for first 3 videos
            print(f"Motion intensity - mean: {motion_intensity.mean():.6f}, max: {motion_intensity.max():.6f}")
        
        return motion_intensity
    
    def apply_patch_attention(self, motion_intensity):
        """Apply patch-based attention mechanism"""
        h, w = motion_intensity.shape
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        
        # Debug: Print dimensions (only for first few videos)
        global patch_debug_counter
        if 'patch_debug_counter' not in globals():
            patch_debug_counter = 0
        patch_debug_counter += 1
        
        if patch_debug_counter <= 3:  # Only print for first 3 videos
            print(f"Motion intensity shape: {motion_intensity.shape}")
            print(f"Patch dimensions: {patch_h} x {patch_w} x {self.patch_size} x {self.patch_size}")
        
        # Reshape to patches
        patches = motion_intensity[:patch_h * self.patch_size, 
                                 :patch_w * self.patch_size].reshape(
            patch_h, self.patch_size, patch_w, self.patch_size
        ).transpose(0, 2, 1, 3)
        
        # Calculate patch attention weights
        patch_means = np.mean(patches, axis=(2, 3))
        patch_vars = np.var(patches, axis=(2, 3))
        
        # Attention based on mean and variance
        attention_weights = patch_means * (1 + patch_vars)
        attention_weights = attention_weights / (attention_weights.max() + 1e-8)
        
        # Apply attention to patches
        attended_patches = patches * attention_weights[:, :, np.newaxis, np.newaxis]
        
        return attended_patches, attention_weights
    
    def generate_spikes(self, attended_patches):
        """Generate binary spike trains with improved thresholding"""
        # Flatten patches
        patch_features = np.mean(attended_patches, axis=(2, 3))
        
        # Debug: Check patch features (only for first few videos)
        global patch_feature_debug_counter
        if 'patch_feature_debug_counter' not in globals():
            patch_feature_debug_counter = 0
        patch_feature_debug_counter += 1
        
        if patch_feature_debug_counter <= 3:  # Only print for first 3 videos
            print(f"Patch features - mean: {patch_features.mean():.6f}, max: {patch_features.max():.6f}")
        
        # Improved adaptive thresholding
        patch_mean = np.mean(patch_features)
        patch_std = np.std(patch_features)
        
        # Use percentile-based threshold for better spike distribution
        threshold_percentile = 25  # Top 25% of patches will spike (more selective for sparsity)
        adaptive_threshold = np.percentile(patch_features, threshold_percentile)
        
        # Ensure minimum threshold with better scaling
        adaptive_threshold = max(adaptive_threshold, patch_mean * 0.4)
        
        # Generate binary spikes
        spike_train = np.where(patch_features > adaptive_threshold, 1.0, 0.0)
        
        # Debug: Check spike train (only for first few videos)
        global spike_debug_counter
        if 'spike_debug_counter' not in globals():
            spike_debug_counter = 0
        spike_debug_counter += 1
        
        if spike_debug_counter <= 3:  # Only print for first 3 videos
            print(f"Spike train - mean: {spike_train.mean():.6f}, sum: {spike_train.sum()}, threshold: {adaptive_threshold:.6f}")
        
        return spike_train, adaptive_threshold
    
    def leaky_integration(self, spike_train):
        """Apply leaky integration to convert spikes to analog voltage"""
        # For spatial features, we'll use a different approach
        # Instead of temporal integration, we'll use spatial smoothing
        
        # Apply Gaussian smoothing to create analog-like output
        
        # Reshape spike train to 2D if needed
        if len(spike_train.shape) == 1:
            # Assume it's a flattened 2D array
            # Try to reshape to square-ish shape
            n_patches = len(spike_train)
            side_length = int(np.sqrt(n_patches))
            
            if side_length * side_length == n_patches:
                # Perfect square
                spike_2d = spike_train.reshape(side_length, side_length)
            else:
                # Not perfect square, pad with zeros
                next_square = (side_length + 1) ** 2
                padded_spikes = np.zeros(next_square)
                padded_spikes[:n_patches] = spike_train
                spike_2d = padded_spikes.reshape(side_length + 1, side_length + 1)
        else:
            spike_2d = spike_train
        
        # Apply Gaussian smoothing for analog-like output
        voltage_output = gaussian_filter(spike_2d.astype(float), sigma=0.5)
        
        # Apply leaky integration (spatial decay)
        voltage_output = voltage_output * self.leak_rate
        
        return voltage_output
    
    def apply_feedforward_inhibition(self, voltage_output):
        """Apply feedforward inhibition to reduce noise"""
        # Simple lateral inhibition
        inhibited_output = voltage_output * (1 - self.feedforward_inhibition)
        
        # Ensure non-negative values
        inhibited_output = np.maximum(inhibited_output, 0)
        
        return inhibited_output
    
    def encode(self, frames):
        """Complete LGMD encoding pipeline"""
        # 1. Extract motion features
        motion_intensity = self.extract_motion_features(frames)
        
        # 2. Apply patch attention
        attended_patches, attention_weights = self.apply_patch_attention(motion_intensity)
        
        # 3. Generate spikes
        spike_train, threshold = self.generate_spikes(attended_patches)
        
        # 4. Leaky integration
        voltage_output = self.leaky_integration(spike_train)
        
        # 5. Apply feedforward inhibition
        final_output = self.apply_feedforward_inhibition(voltage_output)
        
        return final_output

def load_kth_dataset(drive_path="/content/drive/MyDrive/KTH_dataset"):
    """Load KTH dataset from Google Drive"""
    print("Loading KTH dataset...")
    
    # Check if dataset exists
    if not os.path.exists(drive_path):
        print(f"Dataset not found at {drive_path}")
        print("Please upload KTH dataset to Google Drive with the following structure:")
        print("KTH_dataset/")
        print("├── walking/")
        print("├── jogging/")
        print("├── running/")
        print("├── boxing/")
        print("├── waving/")
        print("└── clapping/")
        return None, None
    
    video_data = []
    labels = []
    label_mapping = {
        'walking': 0, 'jogging': 1, 'running': 2,
        'boxing': 3, 'waving': 4, 'clapping': 5
    }
    
    for action in label_mapping.keys():
        action_path = os.path.join(drive_path, action)
        if not os.path.exists(action_path):
            print(f"Warning: {action} folder not found")
            continue
            
        print(f"Loading {action} videos...")
        video_files = glob.glob(os.path.join(action_path, "*.avi"))
        
        for video_file in video_files[:50]:  # Limit to 50 videos per action for speed
            try:
                cap = cv2.VideoCapture(video_file)
                frames = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert to grayscale and resize
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (160, 120))
                    frames.append(resized)
                
                cap.release()
                
                if len(frames) > 0:
                    video_data.append(frames)
                    labels.append(label_mapping[action])
                    
            except Exception as e:
                print(f"Error loading {video_file}: {e}")
                continue
    
    if len(video_data) == 0:
        print("No videos loaded successfully!")
        return None, None
    
    print(f"Loaded {len(video_data)} videos with {len(set(labels))} classes")
    return video_data, labels

def create_synthetic_dataset(num_videos=600, num_frames=20, frame_height=120, frame_width=160):
    """Create synthetic dataset for testing when KTH is not available"""
    print("Creating synthetic dataset for testing...")
    
    video_data = []
    labels = []
    
    # Create 6 different action patterns
    actions = ['walking', 'jogging', 'running', 'boxing', 'waving', 'clapping']
    
    for action_idx, action in enumerate(actions):
        videos_per_action = num_videos // len(actions)
        
        for video_idx in range(videos_per_action):
            frames = []
            
            # Create synthetic motion patterns
            for frame_idx in range(num_frames):
                # Base frame with noise
                frame = np.random.rand(frame_height, frame_width) * 0.3
                
                # Add action-specific motion patterns
                if action == 'walking':
                    # Horizontal motion
                    motion = np.sin(frame_idx * 0.5) * 0.5
                    frame[:, int(frame_width//2 + motion * 20):int(frame_width//2 + motion * 20 + 10)] += 0.7
                    
                elif action == 'jogging':
                    # Faster horizontal motion
                    motion = np.sin(frame_idx * 0.8) * 0.5
                    frame[:, int(frame_width//2 + motion * 15):int(frame_width//2 + motion * 15 + 8)] += 0.8
                    
                elif action == 'running':
                    # Very fast motion
                    motion = np.sin(frame_idx * 1.2) * 0.5
                    frame[:, int(frame_width//2 + motion * 12):int(frame_width//2 + motion * 12 + 6)] += 0.9
                    
                elif action == 'boxing':
                    # Vertical punching motion
                    motion = np.sin(frame_idx * 0.6) * 0.5
                    frame[int(frame_height//2 + motion * 15):int(frame_height//2 + motion * 15 + 8), :] += 0.7
                    
                elif action == 'waving':
                    # Circular motion
                    angle = frame_idx * 0.3
                    center_x, center_y = frame_width//2, frame_height//2
                    x = int(center_x + np.cos(angle) * 20)
                    y = int(center_y + np.sin(angle) * 20)
                    frame[max(0, y-5):min(frame_height, y+5), max(0, x-5):min(frame_width, x+5)] += 0.8
                    
                elif action == 'clapping':
                    # Bilateral motion
                    motion = np.sin(frame_idx * 0.7) * 0.5
                    frame[:, int(frame_width//4 + motion * 10):int(frame_width//4 + motion * 10 + 5)] += 0.6
                    frame[:, int(3*frame_width//4 + motion * 10):int(3*frame_width//4 + motion * 10 + 5)] += 0.6
                
                # Add some temporal consistency
                if frame_idx > 0:
                    frame = 0.8 * frame + 0.2 * frames[-1]
                
                frames.append(frame.astype(np.uint8))
            
            video_data.append(frames)
            labels.append(action_idx)
    
    print(f"Created synthetic dataset: {len(video_data)} videos, {len(set(labels))} classes")
    return video_data, labels

def extract_lgmd_features(video_data, labels, patch_size=8, leak_rate=0.95, 
                         threshold=0.1, feedforward_inhibition=0.3, 
                         directional_weight=0.2):
    """Extract LGMD features from video data"""
    print("Extracting LGMD features...")
    
    encoder = ImprovedLGMDEncoder(
        patch_size=patch_size,
        leak_rate=leak_rate,
        threshold=threshold,
        feedforward_inhibition=feedforward_inhibition,
        directional_weight=directional_weight
    )
    
    features = []
    valid_labels = []
    
    for i, (video, label) in enumerate(zip(video_data, labels)):
        try:
            # Encode video
            feature = encoder.encode(video)
            
            # Flatten feature
            feature_flat = feature.flatten()
            
            # Ensure consistent feature size
            if len(feature_flat) < 100:  # Pad if too small
                feature_flat = np.pad(feature_flat, (0, 100 - len(feature_flat)))
            elif len(feature_flat) > 100:  # Truncate if too large
                feature_flat = feature_flat[:100]
            
            features.append(feature_flat)
            valid_labels.append(label)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(video_data)} videos")
                
        except Exception as e:
            print(f"Error processing video {i}: {e}")
            continue
    
    if len(features) == 0:
        print("No features extracted successfully!")
        return None, None
    
    features = np.array(features)
    valid_labels = np.array(valid_labels)
    
    print(f"Extracted features: {features.shape}")
    return features, valid_labels

def enhance_features(features):
    """Enhance features with additional processing"""
    print("Enhancing features...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Add polynomial features (degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    features_poly = poly.fit_transform(features_scaled)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(100, features_poly.shape[1]))
    features_pca = pca.fit_transform(features_poly)
    
    print(f"Enhanced features: {features_pca.shape}")
    return features_pca

def biologically_plausible_baseline(X, y, test_indices):
    """Biologically plausible baseline using simple classifier"""
    # Split data
    X_train, X_test = X[~test_indices], X[test_indices]
    y_train, y_test = y[~test_indices], y[test_indices]
    
    # Use Ridge regression for biological plausibility
    clf = Ridge(alpha=1.0)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    y_pred_classes = np.round(y_pred).astype(int)
    y_pred_classes = np.clip(y_pred_classes, 0, len(np.unique(y)) - 1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_classes)
    return accuracy

def save_features(features, labels, save_path="/content/drive/MyDrive/lgmd_features.npz"):
    """Save extracted features to Google Drive"""
    try:
        np.savez(save_path, features=features, labels=labels)
        print(f"Features saved to {save_path}")
        return True
    except Exception as e:
        print(f"Error saving features: {e}")
        return False

def load_features(load_path="/content/drive/MyDrive/lgmd_features.npz"):
    """Load saved features from Google Drive"""
    try:
        if os.path.exists(load_path):
            data = np.load(load_path)
            features = data['features']
            labels = data['labels']
            print(f"Features loaded from {load_path}: {features.shape}")
            return features, labels
        else:
            print(f"Features file not found at {load_path}")
            return None, None
    except Exception as e:
        print(f"Error loading features: {e}")
        return None, None

def lgmd_baseline(X, y, test_indices):
    """Simple LGMD baseline"""
    # Split data
    X_train, X_test = X[~test_indices], X[test_indices]
    y_train, y_test = y[~test_indices], y[test_indices]
    
    # Simple nearest neighbor
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def advanced_lgmd_baseline(X, y, test_indices):
    """Advanced LGMD baseline with feature enhancement"""
    # Split data
    X_train, X_test = X[~test_indices], X[test_indices]
    y_train, y_test = y[~test_indices], y[test_indices]
    
    # Enhance features
    X_train_enhanced = enhance_features(X_train)
    X_test_enhanced = enhance_features(X_test)
    
    # Use multiple classifiers
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    
    classifiers = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        SVC(kernel='rbf', random_state=42),
        MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
    ]
    
    accuracies = []
    for clf in classifiers:
        clf.fit(X_train_enhanced, y_train)
        y_pred = clf.predict(X_test_enhanced)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    
    return np.mean(accuracies)

def quick_evaluation(X, y):
    """Quick evaluation of the features"""
    print("Quick evaluation...")
    
    # Use 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    advanced_accuracies = []
    bio_accuracies = []
    
    for train_idx, test_idx in skf.split(X, y):
        # Advanced baseline
        adv_acc = advanced_lgmd_baseline(X, y, test_idx)
        advanced_accuracies.append(adv_acc)
        
        # Biologically plausible baseline
        bio_acc = biologically_plausible_baseline(X, y, test_idx)
        bio_accuracies.append(bio_acc)
    
    mean_advanced = np.mean(advanced_accuracies)
    std_advanced = np.std(advanced_accuracies)
    mean_bio = np.mean(bio_accuracies)
    std_bio = np.std(bio_accuracies)
    
    print(f"Advanced LGMD Baseline: {mean_advanced:.4f} ± {std_advanced:.4f}")
    print(f"Biologically Plausible Baseline: {mean_bio:.4f} ± {std_bio:.4f}")
    
    return mean_advanced, std_advanced, mean_bio, std_bio

def tune_lgmd_parameters(video_data, labels, n_trials=10):
    """Tune LGMD parameters for optimal performance"""
    print("Tuning LGMD parameters for optimal performance...")
    
    import itertools
    
    # Parameter ranges to test
    param_ranges = {
        'patch_size': [6, 8, 10],
        'leak_rate': [0.90, 0.92, 0.95],
        'threshold': [0.05, 0.1, 0.15],
        'feedforward_inhibition': [0.1, 0.15, 0.2],
        'directional_weight': [0.2, 0.3, 0.4]
    }
    
    best_params = None
    best_accuracy = 0
    
    # Generate parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, params in enumerate(param_combinations[:n_trials]):  # Limit trials for speed
        param_dict = dict(zip(param_names, params))
        
        try:
            # Extract features with current parameters
            features, valid_labels = extract_lgmd_features(
                video_data, labels,
                patch_size=param_dict['patch_size'],
                leak_rate=param_dict['leak_rate'],
                threshold=param_dict['threshold'],
                feedforward_inhibition=param_dict['feedforward_inhibition'],
                directional_weight=param_dict['directional_weight']
            )
            
            # Quick evaluation
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            accuracies = []
            
            for train_idx, test_idx in skf.split(features, valid_labels):
                acc = biologically_plausible_baseline(features, valid_labels, test_idx)
                accuracies.append(acc)
            
            mean_acc = np.mean(accuracies)
            
            print(f"Trial {i+1}/{n_trials}: {mean_acc:.4f} - {param_dict}")
            
            if mean_acc > best_accuracy:
                best_accuracy = mean_acc
                best_params = param_dict
                
        except Exception as e:
            print(f"Trial {i+1} failed: {e}")
            continue
    
    print(f"\nBest parameters found: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    
    return best_params, best_accuracy

def quick_parameter_check(video_data, labels):
    """Quick check of current parameters performance"""
    print("Quick parameter performance check...")
    
    # Test current default parameters
    features, valid_labels = extract_lgmd_features(
        video_data, labels,
        patch_size=8,
        leak_rate=0.95,
        threshold=0.1,
        feedforward_inhibition=0.1,
        directional_weight=0.3
    )
    
    # Quick evaluation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accuracies = []
    
    for train_idx, test_idx in skf.split(features, valid_labels):
        acc = biologically_plausible_baseline(features, valid_labels, test_idx)
        accuracies.append(acc)
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"Current parameters performance: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Suggest tuning if performance is low
    if mean_acc < 0.70:
        print("Performance is below 70%. Consider running parameter tuning.")
        return False
    else:
        print("Performance is acceptable. Current parameters work well.")
        return True

def main_part1():
    """Main execution for Part 1: LGMD Feature Extraction"""
    print("=" * 50)
    print("PART 1: LGMD ENCODER & FEATURE EXTRACTION")
    print("=" * 50)
    
    # Mount Google Drive
    if not mount_drive():
        print("Failed to mount Google Drive. Please check your Colab setup.")
        return None, None
    
    # Try to load existing features first
    features, labels = load_features()
    
    # Force re-extraction with new parameters (set to True to re-extract)
    force_re_extract = True
    
    if features is None or force_re_extract:
        if force_re_extract and features is not None:
            print("Forcing re-extraction with new parameters...")
        else:
            print("No existing features found. Loading dataset and extracting features...")
        
        # Try to load KTH dataset first
        video_data, labels = load_kth_dataset()
        
        # If KTH dataset is not available, create synthetic dataset
        if video_data is None:
            print("KTH dataset not found. Creating synthetic dataset for testing...")
            video_data, labels = create_synthetic_dataset()
        
        if video_data is None:
            print("Failed to load or create dataset.")
            return None, None
        
        # Quick parameter check
        print("\n" + "=" * 30)
        print("PARAMETER PERFORMANCE CHECK")
        print("=" * 30)
        performance_ok = quick_parameter_check(video_data, labels)
        
        if not performance_ok:
            print("\n" + "=" * 30)
            print("PARAMETER TUNING SUGGESTED")
            print("=" * 30)
            print("Current parameters show low performance.")
            print("Consider running: tune_lgmd_parameters(video_data, labels)")
            print("This will find optimal parameters for your dataset.")
        
        # Extract LGMD features with optimized parameters
        features, labels = extract_lgmd_features(
            video_data, labels,
            patch_size=8,  # Larger patches for better spatial coherence
            leak_rate=0.95,  # Slower decay for better feature preservation
            threshold=0.1,  # Standard threshold
            feedforward_inhibition=0.1,  # Minimal inhibition to preserve features
            directional_weight=0.3  # Moderate directional selectivity
        )
        
        # Save features for future use
        save_features(features, labels)
    else:
        print("Existing features loaded successfully!")
    
    # Quick evaluation
    mean_advanced, std_advanced, mean_bio, std_bio = quick_evaluation(features, labels)
    
    print("\n" + "=" * 50)
    print("PART 1 COMPLETED")
    print("=" * 50)
    print(f"Features ready for Part 2: {features.shape}")
    print(f"Advanced LGMD Baseline Performance: {mean_advanced:.4f} ± {std_advanced:.4f}")
    print(f"Biologically Plausible LGMD Baseline Performance: {mean_bio:.4f} ± {std_bio:.4f}")
    
    return features, labels

# Run the main function
if __name__ == "__main__":
    features, labels = main_part1() 