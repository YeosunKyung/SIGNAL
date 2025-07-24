import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter
import time
import warnings
import os
from collections import Counter
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import geoopt
from geoopt.manifolds import PoincareBall

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class ImprovedLGMDEncoder:
    """
    생물학적으로 현실적인 LGMD encoder with Topological Singularity Detection
    - 스파이크는 바이너리로 생성하되
    - 출력은 leaky integration된 아날로그 전압값
    - NEW: Topological singularity detection for motion patterns
    """
    
    def __init__(self, patch_size=8, leak_rate=0.95, threshold=0.1, 
                 feedforward_inhibition=0.3, directional_weight=0.2,
                 singularity_detection=True, singularity_weight=0.3):
        self.patch_size = patch_size
        self.leak_rate = leak_rate
        self.threshold = threshold
        self.feedforward_inhibition = feedforward_inhibition
        self.directional_weight = directional_weight
        self.singularity_detection = singularity_detection
        self.singularity_weight = singularity_weight
        self.singularities = None  # Store detected singularities
        
    def detect_motion_singularities(self, motion_features):
        """
        Detect topological singularities in motion patterns
        Based on critical point analysis and flow topology
        """
        # Smooth the motion field for stable gradient computation
        motion_smooth = gaussian_filter(motion_features, sigma=1.0)
        
        # Compute gradients
        grad_y, grad_x = np.gradient(motion_smooth)
        
        # Compute second derivatives (Hessian components)
        grad_yy, grad_yx = np.gradient(grad_y)
        grad_xy, grad_xx = np.gradient(grad_x)
        
        # Hessian determinant and trace for critical point classification
        det_H = grad_xx * grad_yy - grad_xy * grad_yx
        trace_H = grad_xx + grad_yy
        
        # Eigenvalues analysis
        # λ1,2 = (trace ± sqrt(trace² - 4*det)) / 2
        discriminant = trace_H**2 - 4*det_H
        
        # Classify singularities
        # Saddle points: det < 0 (opposing flow)
        saddle_points = (det_H < -0.01)
        
        # Sources/sinks: det > 0, trace != 0
        sources = (det_H > 0.01) & (trace_H > 0.01)  # Diverging flow
        sinks = (det_H > 0.01) & (trace_H < -0.01)   # Converging flow
        
        # Vortices: det > 0, trace ≈ 0 (rotational flow)
        vortices = (det_H > 0.01) & (np.abs(trace_H) < 0.01)
        
        # Combined singularity map
        singularity_map = saddle_points | sources | sinks | vortices
        
        # Compute singularity strength (importance)
        singularity_strength = np.sqrt(det_H**2 + trace_H**2)
        singularity_strength[~singularity_map] = 0
        
        return {
            'saddle': saddle_points,
            'sources': sources,
            'sinks': sinks,
            'vortices': vortices,
            'combined_map': singularity_map,
            'strength': singularity_strength,
            'locations': np.column_stack(np.where(singularity_map))
        }
    
    def extract_motion_features(self, frames):
        """Motion intensity extraction with directional selectivity"""
        if len(frames) < 2:
            return np.zeros((frames[0].shape[0] // self.patch_size, 
                           frames[0].shape[1] // self.patch_size))
        
        motion_features = []
        for i in range(1, len(frames)):
            # Frame difference
            diff = cv2.absdiff(frames[i], frames[i-1])
            
            # Directional derivatives (horizontal and vertical)
            grad_x = cv2.Sobel(diff, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(diff, cv2.CV_64F, 0, 1, ksize=3)
            
            # Directional motion intensity
            directional_motion = np.sqrt(grad_x**2 + grad_y**2)
            
            # Combine with regular motion
            combined_motion = (1 - self.directional_weight) * diff + \
                            self.directional_weight * directional_motion
            
            motion_features.append(combined_motion)
        
        return np.mean(motion_features, axis=0)
    
    def apply_patch_attention_with_singularities(self, motion_intensity, singularities=None):
        """Apply patch-based attention mechanism enhanced by singularity information"""
        h, w = motion_intensity.shape
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        
        # Reshape to patches
        patches = motion_intensity[:patch_h * self.patch_size, 
                                 :patch_w * self.patch_size].reshape(
            patch_h, self.patch_size, patch_w, self.patch_size
        ).transpose(0, 2, 1, 3)
        
        # Calculate patch attention weights
        patch_means = np.mean(patches, axis=(2, 3))
        patch_vars = np.var(patches, axis=(2, 3))
        
        # Basic attention based on mean and variance
        attention_weights = patch_means * (1 + patch_vars)
        
        # Enhance attention at singularity locations
        if singularities is not None and self.singularity_detection:
            # Resize singularity strength map to patch size
            sing_strength = cv2.resize(singularities['strength'], 
                                     (patch_w, patch_h), 
                                     interpolation=cv2.INTER_AREA)
            
            # Boost attention at singularity patches
            attention_weights = attention_weights * (1 + self.singularity_weight * sing_strength)
        
        # Normalize
        attention_weights = attention_weights / (attention_weights.max() + 1e-8)
        
        # Apply attention to patches
        attended_patches = patches * attention_weights[:, :, np.newaxis, np.newaxis]
        
        return attended_patches, attention_weights
    
    def generate_spikes(self, attended_patches):
        """Generate binary spike trains"""
        # Flatten patches
        patch_features = np.mean(attended_patches, axis=(2, 3))
        
        # Adaptive thresholding based on input intensity
        patch_mean = np.mean(patch_features)
        adaptive_threshold = max(self.threshold, patch_mean * 0.3)
        
        # Generate binary spikes
        spike_train = np.where(patch_features > adaptive_threshold, 1.0, 0.0)
        
        return spike_train, adaptive_threshold
    
    def leaky_integration(self, spike_train):
        """Apply leaky integration to convert spikes to analog voltage"""
        voltage_output = np.zeros_like(spike_train)
        
        for t in range(1, spike_train.shape[0]):
            voltage_output[t] = self.leak_rate * voltage_output[t-1] + spike_train[t]
        
        # Normalize to 0-1 range
        if voltage_output.max() > 0:
            voltage_output = voltage_output / voltage_output.max()
        
        return voltage_output
    
    def apply_feedforward_inhibition(self, voltage_output):
        """Apply feedforward lateral inhibition"""
        h, w = voltage_output.shape
        
        # Lateral inhibition kernel
        inhibition_kernel = np.array([
            [0, -self.feedforward_inhibition, 0],
            [-self.feedforward_inhibition, 1, -self.feedforward_inhibition],
            [0, -self.feedforward_inhibition, 0]
        ])
        
        # Apply inhibition (with padding)
        inhibited_output = np.zeros_like(voltage_output)
        for i in range(1, h-1):
            for j in range(1, w-1):
                patch = voltage_output[i-1:i+2, j-1:j+2]
                inhibited_output[i, j] = np.sum(patch * inhibition_kernel)
        
        # Ensure non-negative values
        inhibited_output = np.maximum(inhibited_output, 0)
        
        return inhibited_output
    
    def encode_with_singularities(self, frames):
        """Complete LGMD encoding process with singularity preservation"""
        if len(frames) == 0:
            return np.array([]), None
        
        # 1. Motion feature extraction with directional selectivity
        motion_intensity = self.extract_motion_features(frames)
        
        # 2. Detect topological singularities
        if self.singularity_detection:
            self.singularities = self.detect_motion_singularities(motion_intensity)
        else:
            self.singularities = None
        
        # 3. Patch-based attention enhanced by singularities
        attended_patches, attention_weights = self.apply_patch_attention_with_singularities(
            motion_intensity, self.singularities
        )
        
        # 4. Generate spike trains
        spike_train, adaptive_threshold = self.generate_spikes(attended_patches)
        
        # 5. Leaky integration to analog voltage
        voltage_output = self.leaky_integration(spike_train)
        
        # 6. Feedforward inhibition
        inhibited_output = self.apply_feedforward_inhibition(voltage_output)
        
        # 7. Create feature vector with singularity information
        base_features = inhibited_output.flatten()
        
        if self.singularity_detection and self.singularities is not None:
            # Add singularity-derived features
            sing_features = []
            
            # Number of each type of singularity
            sing_features.append(np.sum(self.singularities['saddle']))
            sing_features.append(np.sum(self.singularities['sources']))
            sing_features.append(np.sum(self.singularities['sinks']))
            sing_features.append(np.sum(self.singularities['vortices']))
            
            # Average strength of singularities
            if len(self.singularities['locations']) > 0:
                sing_features.append(np.mean(self.singularities['strength'][self.singularities['combined_map']]))
            else:
                sing_features.append(0)
            
            # Spatial distribution features
            if len(self.singularities['locations']) > 0:
                locations = self.singularities['locations']
                sing_features.append(np.std(locations[:, 0]))  # Vertical spread
                sing_features.append(np.std(locations[:, 1]))  # Horizontal spread
            else:
                sing_features.extend([0, 0])
            
            singularity_vector = np.array(sing_features)
            
            # Combine base features with singularity features
            feature_vector = np.concatenate([base_features, singularity_vector])
        else:
            feature_vector = base_features
        
        # Debug prints
        print(f"LGMD Debug - Spike train mean: {spike_train.mean():.4f}, "
              f"Voltage mean: {voltage_output.mean():.4f}, "
              f"Final feature mean: {feature_vector.mean():.4f}")
        
        if self.singularity_detection and self.singularities is not None:
            print(f"Singularities detected - Saddles: {np.sum(self.singularities['saddle'])}, "
                  f"Sources: {np.sum(self.singularities['sources'])}, "
                  f"Sinks: {np.sum(self.singularities['sinks'])}, "
                  f"Vortices: {np.sum(self.singularities['vortices'])}")
        
        return feature_vector, self.singularities
    
    def encode(self, frames):
        """Wrapper for backward compatibility"""
        features, _ = self.encode_with_singularities(frames)
        return features

# Topological Dimension Reduction class
class TopologicalDimensionReduction:
    """Dimension reduction that preserves topological singularities"""
    
    def __init__(self, target_dim=20, preserve_ratio=0.8, use_svd=True):
        self.target_dim = target_dim
        self.preserve_ratio = preserve_ratio
        self.use_svd = use_svd
        
    def fit_transform(self, features, singularity_info=None):
        """Reduce dimensions while preserving singularity structure"""
        n_samples, n_features = features.shape
        
        if singularity_info is None:
            # Fallback to standard PCA
            return PCA(n_components=self.target_dim).fit_transform(features)
        
        # Extract importance weights based on singularities
        importance_weights = self._compute_importance_weights(features, singularity_info)
        
        if self.use_svd:
            # SVD-based reduction with weighted features
            weighted_features = features * np.sqrt(importance_weights)
            U, S, Vt = np.linalg.svd(weighted_features, full_matrices=False)
            
            # Select components that preserve singularity information
            n_preserve = int(self.target_dim * self.preserve_ratio)
            
            # Find components with highest singularity correlation
            singularity_scores = []
            for i in range(min(len(S), features.shape[1])):
                component = Vt[i, :]
                score = np.sum(component**2 * importance_weights)
                singularity_scores.append(score)
            
            # Select top components
            top_indices = np.argsort(singularity_scores)[-self.target_dim:]
            
            # Project data
            reduced_features = U[:, top_indices] @ np.diag(S[top_indices])
        else:
            # Direct feature selection based on importance
            top_feature_indices = np.argsort(importance_weights)[-self.target_dim:]
            reduced_features = features[:, top_feature_indices]
        
        return reduced_features
    
    def _compute_importance_weights(self, features, singularity_info):
        """Compute feature importance based on singularity information"""
        n_features = features.shape[1]
        weights = np.ones(n_features)
        
        # Boost weights for features near singularities
        if 'strength' in singularity_info:
            # Flatten singularity strength map
            sing_strength_flat = singularity_info['strength'].flatten()
            
            # Match dimensions
            if len(sing_strength_flat) == n_features:
                weights += sing_strength_flat
            else:
                # Interpolate to match feature dimensions
                weights += np.interp(
                    np.linspace(0, 1, n_features),
                    np.linspace(0, 1, len(sing_strength_flat)),
                    sing_strength_flat
                )
        
        # Normalize weights
        weights = weights / (weights.max() + 1e-8)
        
        return weights

def extract_lgmd_features(video_data, labels, patch_size=8, leak_rate=0.95, 
                         threshold=0.1, feedforward_inhibition=0.3, 
                         directional_weight=0.2, use_singularity=True,
                         use_topological_reduction=True, target_dim=100):
    """
    Extract LGMD features from video data using improved encoder with singularity detection
    """
    print("Extracting LGMD features with improved encoder (singularity detection enabled)...")
    
    encoder = ImprovedLGMDEncoder(
        patch_size=patch_size,
        leak_rate=leak_rate,
        threshold=threshold,
        feedforward_inhibition=feedforward_inhibition,
        directional_weight=directional_weight,
        singularity_detection=use_singularity
    )
    
    features = []
    valid_labels = []
    failed_count = 0
    all_singularities = []
    
    for i, (video, label) in enumerate(zip(video_data, labels)):
        try:
            # Convert video to grayscale frames
            frames = []
            for frame in video:
                if len(frame.shape) == 3:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray_frame = frame
                frames.append(gray_frame.astype(np.float32) / 255.0)
            
            # Extract features with singularities
            feature, singularities = encoder.encode_with_singularities(frames)
            
            if len(feature) > 0 and not np.any(np.isnan(feature)):
                features.append(feature)
                valid_labels.append(label)
                if singularities is not None:
                    all_singularities.append(singularities)
            else:
                failed_count += 1
                print(f"Failed to extract features for video {i}")
                
        except Exception as e:
            failed_count += 1
            print(f"Error processing video {i}: {e}")
    
    if len(features) == 0:
        raise ValueError("No valid features extracted!")
    
    features = np.array(features)
    valid_labels = np.array(valid_labels)
    
    print(f"LGMD feature extraction: {len(features)} succeeded, {failed_count} failed.")
    print(f"Original feature shape: {features.shape}, Labels: {valid_labels.shape}")
    
    # Apply topological dimension reduction if requested
    if use_topological_reduction and use_singularity:
        print(f"Applying topological dimension reduction to {target_dim} dimensions...")
        tdr = TopologicalDimensionReduction(target_dim=target_dim)
        
        # Use average singularity info for dimension reduction
        avg_singularity_info = _average_singularity_info(all_singularities)
        features = tdr.fit_transform(features, avg_singularity_info)
        
        print(f"Reduced feature shape: {features.shape}")
    
    # Feature statistics
    print(f"Feature mean: {features.mean():.6f} std: {features.std():.6f}")
    print(f"Feature min/max: {features.min():.3f} {features.max():.3f}")
    print(f"NaN/Inf count: {np.isnan(features).sum()} {np.isinf(features).sum()}")
    
    return features, valid_labels

def _average_singularity_info(singularity_list):
    """Compute average singularity information across all samples"""
    if not singularity_list:
        return None
    
    # Average the strength maps
    strength_maps = [s['strength'] for s in singularity_list if 'strength' in s]
    if strength_maps:
        avg_strength = np.mean(strength_maps, axis=0)
    else:
        avg_strength = np.zeros_like(singularity_list[0]['combined_map'])
    
    return {
        'strength': avg_strength,
        'combined_map': avg_strength > 0
    }

# Modified HyperbolicContrastiveEmbedding to work with singularity-enhanced features
class HyperbolicContrastiveEmbedding:
    """Hyperbolic contrastive embedding with Poincaré ball"""
    
    def __init__(self, embed_dim=64, curvature=1.0, temperature=0.1, 
                 use_singularity_weighting=True):
        self.embed_dim = embed_dim
        self.curvature = curvature
        self.temperature = temperature
        self.manifold = PoincareBall(c=curvature)
        self.use_singularity_weighting = use_singularity_weighting
        
    def fit_transform(self, X, y):
        """Fit and transform features to hyperbolic space"""
        print("Computing hyperbolic embeddings...")
        
        # Initialize embeddings randomly on Poincaré ball
        embeddings = torch.randn(len(X), self.embed_dim) * 0.1
        embeddings = self.manifold.projx(embeddings)
        
        # Detect if features contain singularity information
        feature_dim = X.shape[1]
        has_singularity_features = feature_dim > 1000  # Heuristic check
        
        # Contrastive learning
        optimizer = optim.Adam([embeddings.requires_grad_()], lr=0.01)
        
        for epoch in range(100):
            loss = 0
            for i in range(len(X)):
                # Positive pairs (same class)
                pos_indices = np.where(y == y[i])[0]
                pos_indices = pos_indices[pos_indices != i]
                
                if len(pos_indices) > 0:
                    pos_idx = np.random.choice(pos_indices)
                    pos_dist = self.manifold.dist(embeddings[i], embeddings[pos_idx])
                    
                    # Negative pairs (different class)
                    neg_indices = np.where(y != y[i])[0]
                    if len(neg_indices) > 0:
                        neg_idx = np.random.choice(neg_indices)
                        neg_dist = self.manifold.dist(embeddings[i], embeddings[neg_idx])
                        
                        # Weight by singularity importance if available
                        if self.use_singularity_weighting and has_singularity_features:
                            # Use last few features as singularity indicators
                            sing_weight_i = 1 + np.sum(X[i, -7:]) / 7  # Last 7 features are singularity-based
                            sing_weight_pos = 1 + np.sum(X[pos_idx, -7:]) / 7
                            weight = (sing_weight_i + sing_weight_pos) / 2
                        else:
                            weight = 1.0
                        
                        # Weighted contrastive loss
                        loss += weight * torch.clamp(pos_dist - neg_dist + self.temperature, min=0)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Project back to manifold
            with torch.no_grad():
                embeddings.data = self.manifold.projx(embeddings.data)
        
        return embeddings.detach().numpy(), self.manifold

# Rest of the functions remain the same...
def hyperbolic_structural_plasticity(Z_poincare, y, manifold, novelty_thresh=0.1, 
                                   redundancy_thresh=0.3, max_prototypes_per_class=5):
    """Structural plasticity for prototype selection in hyperbolic space"""
    print("Applying structural plasticity...")
    
    Z_poincare_np = Z_poincare.numpy() if torch.is_tensor(Z_poincare) else Z_poincare
    unique_classes = np.unique(y)
    selected_prototypes = []
    prototype_labels = []
    
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        class_embeddings = Z_poincare_np[class_indices]
        
        # Initialize with first sample
        prototypes = [class_embeddings[0]]
        prototype_labels.append(class_label)
        
        for i in range(1, len(class_embeddings)):
            embedding = class_embeddings[i]
            
            # Check novelty (distance to existing prototypes)
            min_dist_to_prototypes = float('inf')
            for proto in prototypes:
                dist = manifold.dist(torch.tensor(embedding), torch.tensor(proto)).item()
                min_dist_to_prototypes = min(min_dist_to_prototypes, dist)
            
            # Check redundancy (distance to other samples in class)
            distances_to_class = []
            for j, other_embedding in enumerate(class_embeddings):
                if i != j:
                    dist = manifold.dist(torch.tensor(embedding), torch.tensor(other_embedding)).item()
                    distances_to_class.append(dist)
            
            avg_dist_to_class = np.mean(distances_to_class) if distances_to_class else 0
            
            # Novelty and redundancy criteria
            is_novel = min_dist_to_prototypes > novelty_thresh
            is_not_redundant = avg_dist_to_class > redundancy_thresh
            
            if is_novel and is_not_redundant and len(prototypes) < max_prototypes_per_class:
                prototypes.append(embedding)
                prototype_labels.append(class_label)
        
        selected_prototypes.extend(prototypes)
    
    if len(selected_prototypes) == 0:
        print("Warning: No prototypes found after structural plasticity. Using class means as fallback.")
        # Fallback: use class means
        for class_label in unique_classes:
            class_indices = np.where(y == class_label)[0]
            class_embeddings = Z_poincare_np[class_indices]
            class_mean = np.mean(class_embeddings, axis=0)
            selected_prototypes.append(class_mean)
            prototype_labels.append(class_label)
    
    print(f"Selected {len(selected_prototypes)} prototypes from {len(unique_classes)} classes")
    return np.array(selected_prototypes), np.array(prototype_labels)

# Continue with the rest of the original code...
class SemanticDecoder:
    """Semantic decoder for final classification"""
    
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def fit(self, X, y, epochs=100, lr=0.001):
        """Train the semantic decoder"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == y_tensor).float().mean().item()
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
    
    def predict(self, X):
        """Predict class labels"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.numpy()

# Modified ablation study to include singularity analysis
def ablation_study_with_singularity(X, y, video_data, labels):
    """Ablation study including singularity detection impact"""
    print("\n" + "=" * 50)
    print("ABLATION STUDY WITH SINGULARITY ANALYSIS")
    print("=" * 50)
    
    # Test configurations
    configs = [
        {'name': 'Full Model (with Singularity + Topological Reduction)', 
         'use_singularity': True, 'use_topological_reduction': True, 'target_dim': 100},
        {'name': 'Without Topological Reduction', 
         'use_singularity': True, 'use_topological_reduction': False, 'target_dim': None},
        {'name': 'Without Singularity Detection', 
         'use_singularity': False, 'use_topological_reduction': False, 'target_dim': None},
        {'name': 'Baseline LGMD', 
         'use_singularity': False, 'use_topological_reduction': False, 'target_dim': None}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        # Extract features with current configuration
        X_config, y_config = extract_lgmd_features(
            video_data, labels,
            use_singularity=config['use_singularity'],
            use_topological_reduction=config['use_topological_reduction'],
            target_dim=config['target_dim'] if config['target_dim'] else 100
        )
        
        # Quick evaluation
        test_size = min(100, len(X_config) // 3)
        test_indices = np.random.choice(len(X_config), test_size, replace=False)
        
        # Train and evaluate
        from sklearn.svm import SVC
        clf = SVC(kernel='rbf', random_state=42)
        
        train_mask = np.ones(len(X_config), dtype=bool)
        train_mask[test_indices] = False
        
        clf.fit(X_config[train_mask], y_config[train_mask])
        y_pred = clf.predict(X_config[test_indices])
        acc = accuracy_score(y_config[test_indices], y_pred)
        
        results.append({
            'config': config['name'],
            'accuracy': acc,
            'feature_dim': X_config.shape[1]
        })
        
        print(f"Accuracy: {acc:.3f}, Feature dimension: {X_config.shape[1]}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("ABLATION SUMMARY")
    print("=" * 50)
    for result in results:
        print(f"{result['config']}: {result['accuracy']:.3f} (dim={result['feature_dim']})")
    
    return results

# The main function and other baseline functions remain the same as in the original code...

def main():
    """Main pipeline execution with singularity detection"""
    print("IMPROVED LGMD HYPERBOLIC PIPELINE WITH TOPOLOGICAL SINGULARITY DETECTION")
    print("=" * 70)
    
    # Load and preprocess KTH dataset
    print("Loading KTH dataset...")
    
    # For demonstration, create synthetic video data
    # In practice, load actual KTH dataset
    num_videos = 600
    num_frames = 20
    frame_height, frame_width = 120, 160
    
    # Create synthetic video data (replace with actual KTH loading)
    video_data = []
    labels = []
    
    for i in range(num_videos):
        # Create synthetic video frames
        video = []
        for frame_idx in range(num_frames):
            # Create frame with some motion pattern
            frame = np.random.rand(frame_height, frame_width, 3)
            if frame_idx > 0:
                # Add some motion between frames
                motion = np.random.rand(frame_height, frame_width, 3) * 0.1
                frame = np.clip(frame + motion, 0, 1)
            video.append(frame)
        
        video_data.append(video)
        labels.append(i % 6)  # 6 classes
    
    video_data = np.array(video_data)
    labels = np.array(labels)
    
    print(f"Dataset loaded: {len(video_data)} videos, {len(np.unique(labels))} classes")
    
    # Extract LGMD features with singularity detection
    X, y = extract_lgmd_features(
        video_data, labels,
        patch_size=8,
        leak_rate=0.95,
        threshold=0.1,
        feedforward_inhibition=0.3,
        directional_weight=0.2,
        use_singularity=True,
        use_topological_reduction=True,
        target_dim=100  # Reduced dimension
    )
    
    print(f"X shape: {X.shape} y shape: {y.shape}")
    print(f"First 10 labels: {y[:10]}")
    
    # Robust evaluation
    results = robust_evaluation(X, y)
    
    # Statistical significance testing
    statistical_significance_testing(results)
    
    # Ablation study with singularity analysis
    ablation_results = ablation_study_with_singularity(X, y, video_data, labels)
    
    # Hyperparameter sweep
    hyperparameter_sweep(X, y)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED WITH SINGULARITY DETECTION")
    print("=" * 70)

# Keep all other functions (cnn_baseline, snn_baseline, lgmd_baseline, 
# robust_evaluation, statistical_significance_testing, hyperparameter_sweep) 
# from the original code as they are...

if __name__ == "__main__":
    main()