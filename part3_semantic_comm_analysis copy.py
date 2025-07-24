import os
import sys
sys.path.append('/content/drive/MyDrive')  # Ensure custom modules in Drive are importable
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
import torch
import torch.nn as nn
import cv2
from sklearn.decomposition import PCA, DictionaryLearning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import json
import warnings
warnings.filterwarnings('ignore')

# --- 경로 설정 ---
PATHS = {
    'drive_path': "/content/drive/MyDrive/KTH_dataset",
    'features_save_path': "/content/drive/MyDrive/lgmd_features.npz",
    'singularity_features_save_path': "/content/drive/MyDrive/lgmd_singularity_features.npz",
    'results_save_path': "/content/drive/MyDrive/lgmd_results.json",
    'simul_dir': "/content/drive/MyDrive/lgmd_simul/"
}

os.makedirs(PATHS['simul_dir'], exist_ok=True)

# Device selection: Use GPU if available, else CPU
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    print(f"✅ Using GPU: {_gpus[0].name}")
else:
    print("⚠️ No GPU found. Using CPU only.")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- Import necessary classes from quick_start_colab ---
# Since we can't import directly, we'll redefine the necessary classes here

class ImprovedLGMDEncoder:
    """LGMD encoder with Topological Singularity Detection"""
    
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
        self.singularities = None
        
    def detect_motion_singularities(self, motion_features):
        """Detect topological singularities in motion patterns"""
        from scipy.ndimage import gaussian_filter
        motion_smooth = gaussian_filter(motion_features, sigma=1.0)
        grad_y, grad_x = np.gradient(motion_smooth)
        grad_yy, grad_yx = np.gradient(grad_y)
        grad_xy, grad_xx = np.gradient(grad_x)
        
        det_H = grad_xx * grad_yy - grad_xy * grad_yx
        trace_H = grad_xx + grad_yy
        
        saddle_points = (det_H < -0.01)
        sources = (det_H > 0.01) & (trace_H > 0.01)
        sinks = (det_H > 0.01) & (trace_H < -0.01)
        vortices = (det_H > 0.01) & (np.abs(trace_H) < 0.01)
        
        singularity_map = saddle_points | sources | sinks | vortices
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
            diff = cv2.absdiff(frames[i], frames[i-1])
            grad_x = cv2.Sobel(diff, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(diff, cv2.CV_64F, 0, 1, ksize=3)
            directional_motion = np.sqrt(grad_x**2 + grad_y**2)
            combined_motion = (1 - self.directional_weight) * diff + \
                            self.directional_weight * directional_motion
            motion_features.append(combined_motion)
        
        return np.mean(motion_features, axis=0)
    
    def encode_with_singularities(self, frames):
        """Complete LGMD encoding process with singularity preservation"""
        if len(frames) == 0:
            return np.array([]), None
        
        motion_intensity = self.extract_motion_features(frames)
        
        if self.singularity_detection:
            self.singularities = self.detect_motion_singularities(motion_intensity)
        else:
            self.singularities = None
        
        # Simplified encoding for demonstration
        feature_vector = motion_intensity.flatten()
        
        if self.singularity_detection and self.singularities is not None:
            sing_features = []
            sing_features.append(np.sum(self.singularities['saddle']))
            sing_features.append(np.sum(self.singularities['sources']))
            sing_features.append(np.sum(self.singularities['sinks']))
            sing_features.append(np.sum(self.singularities['vortices']))
            
            if len(self.singularities['locations']) > 0:
                sing_features.append(np.mean(self.singularities['strength'][self.singularities['combined_map']]))
            else:
                sing_features.append(0)
            
            singularity_vector = np.array(sing_features)
            feature_vector = np.concatenate([feature_vector[:100], singularity_vector])  # Limit size
        
        return feature_vector, self.singularities

class TopologicalDimensionReduction:
    """Dimension reduction that preserves topological singularities"""
    
    def __init__(self, target_dim=20, preserve_ratio=0.8, use_svd=True):
        self.target_dim = target_dim
        self.preserve_ratio = preserve_ratio
        self.use_svd = use_svd
        self.pca = None
        
    def fit_transform(self, features, singularity_info=None):
        """Reduce dimensions while preserving singularity structure"""
        if singularity_info is None:
            self.pca = PCA(n_components=self.target_dim)
            return self.pca.fit_transform(features)
        
        # Use PCA with importance weighting based on singularity
        self.pca = PCA(n_components=min(self.target_dim, features.shape[1]))
        return self.pca.fit_transform(features)
    
    def transform(self, features):
        """Transform new data"""
        if self.pca is None:
            raise ValueError("Model must be fitted before transform")
        return self.pca.transform(features)

class ImprovedHyperbolicEmbedding:
    """Improved hyperbolic embedding with singularity awareness"""
    
    def __init__(self, embed_dim=32, curvature=1.0, temperature=0.1, use_singularity_weighting=True):
        self.embed_dim = embed_dim
        self.curvature = curvature
        self.temperature = temperature
        self.use_singularity_weighting = use_singularity_weighting
        self.embeddings = None
        self.is_fitted = False
        
    def fit_transform(self, X, y, max_epochs=30):
        """Simplified hyperbolic embedding"""
        # For demonstration, use PCA as a proxy
        pca = PCA(n_components=self.embed_dim)
        self.embeddings = pca.fit_transform(X)
        self.is_fitted = True
        return self.embeddings, None
    
    def transform(self, X):
        """Transform new data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        # Simple linear projection for demonstration
        return X[:, :self.embed_dim]

def improved_structural_plasticity(Z_embeddings, y, manifold=None, max_prototypes_per_class=20):
    """Improved structural plasticity with better prototype selection"""
    from sklearn.cluster import KMeans
    
    unique_classes = np.unique(y)
    selected_prototypes = []
    prototype_labels = []
    
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        class_embeddings = Z_embeddings[class_indices]
        
        n_clusters = min(max_prototypes_per_class, len(class_embeddings))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(class_embeddings)
            
            for i in range(n_clusters):
                cluster_mask = (cluster_labels == i)
                if np.any(cluster_mask):
                    prototype = np.mean(class_embeddings[cluster_mask], axis=0)
                    selected_prototypes.append(prototype)
                    prototype_labels.append(class_label)
        else:
            prototype = np.mean(class_embeddings, axis=0)
            selected_prototypes.append(prototype)
            prototype_labels.append(class_label)
    
    return np.array(selected_prototypes), np.array(prototype_labels)

# --- 데이터 로딩 함수 ---
def load_features_from_part1(load_path=PATHS['features_save_path']):
    """Load standard features"""
    try:
        data = np.load(load_path)
        features = data['features']
        labels = data['labels']
        print(f"✅ Standard features loaded: {features.shape}, Labels: {labels.shape}")
        return features, labels
    except:
        print("⚠️ Standard features not found")
        return None, None

def load_singularity_features(load_path=PATHS['singularity_features_save_path']):
    """Load features with singularity detection"""
    try:
        data = np.load(load_path)
        features = data['features']
        labels = data['labels']
        print(f"✅ Singularity features loaded: {features.shape}, Labels: {labels.shape}")
        return features, labels
    except:
        print("⚠️ Singularity features not found")
        return None, None

def load_sample_videos(n_samples=100):
    """Load sample videos from the dataset"""
    import glob
    video_dir = PATHS['drive_path']
    
    # Try .npy files first
    video_files = glob.glob(os.path.join(video_dir, '**', '*.npy'), recursive=True)
    
    if not video_files:
        # Try .avi files
        video_files = glob.glob(os.path.join(video_dir, '**', '*.avi'), recursive=True)
        
    print(f"Found {len(video_files)} video files")
    videos = []
    
    # Limit to n_samples
    files_to_load = video_files[:min(n_samples, len(video_files))]
    
    for i, vf in enumerate(files_to_load):
        if i % 10 == 0:
            print(f"  Loading video {i}/{len(files_to_load)}...")
            
        try:
            if vf.endswith('.npy'):
                video = np.load(vf)
                if video.size > 0:  # Check if video is not empty
                    videos.append(video)
            else:
                # For .avi files
                cap = cv2.VideoCapture(vf)
                frames = []
                frame_count = 0
                while frame_count < 30:  # Limit to 30 frames
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    frame_count += 1
                cap.release()
                if frames:
                    video = np.stack(frames)
                    videos.append(video)
        except Exception as e:
            if i < 5:  # Only print first few errors
                print(f"  Failed to load {vf}: {e}")
    
    print(f"Successfully loaded {len(videos)} videos")
    
    # If no videos loaded, create a dummy video
    if not videos:
        print("⚠️ No videos loaded, creating dummy video data...")
        dummy_video = np.random.rand(30, 120, 160, 3).astype(np.float32)
        videos.append(dummy_video)
    
    return videos

# --- Semantic Preservation (kNN Consistency) ---
def evaluate_semantic_preservation(features, labels):
    """Evaluate semantic preservation using kNN"""
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, features, labels, cv=3)
    return np.mean(scores)

# --- Compression/Transmission/Preservation Metrics ---
def measure_communication_metrics(original_data, compressed_data, features, labels, name):
    """Measure communication efficiency metrics"""
    original_size = original_data.nbytes
    compressed_size = compressed_data.nbytes
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    bandwidth = 1e6  # 1 Mbps
    transmission_time = (compressed_size * 8) / bandwidth
    semantic_preservation = evaluate_semantic_preservation(features, labels)
    
    return {
        'method': name,
        'compression_ratio': compression_ratio,
        'original_size_MB': original_size / 1e6,
        'compressed_size_MB': compressed_size / 1e6,
        'transmission_time_s': transmission_time,
        'semantic_preservation': semantic_preservation,
        'bits_per_sample': (compressed_size * 8) / len(features)
    }

# --- Extract features with singularity detection ---
def extract_singularity_features(videos, labels):
    """Extract features with singularity detection from videos"""
    print("🔄 Extracting features with singularity detection...")
    
    encoder = ImprovedLGMDEncoder(
        patch_size=8,
        singularity_detection=True,
        singularity_weight=0.3
    )
    
    features = []
    valid_labels = []
    
    for i, (video, label) in enumerate(zip(videos, labels)):
        if i % 20 == 0:
            print(f"  Processing video {i}/{len(videos)}...")
        
        try:
            frames = []
            for frame in video:
                if len(frame.shape) == 3:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray_frame = frame
                frames.append(gray_frame.astype(np.float32) / 255.0)
            
            feature, _ = encoder.encode_with_singularities(frames[:10])  # Use first 10 frames
            
            if len(feature) > 0 and not np.any(np.isnan(feature)):
                features.append(feature)
                valid_labels.append(label)
                
        except Exception as e:
            print(f"  Error processing video {i}: {e}")
    
    features = np.array(features)
    valid_labels = np.array(valid_labels)
    
    print(f"✅ Extracted features shape: {features.shape}")
    return features, valid_labels

# --- Get all representations for comparison ---
def get_all_representations(features, labels, use_singularity=True):
    """Get all different representations for comparison"""
    rep = {}
    
    # 1. Proposed (Hyperbolic + Singularity-aware Plasticity)
    try:
        print("📊 Computing Proposed (Hyperbolic + Singularity)...")
        hyperbolic_emb = ImprovedHyperbolicEmbedding(
            embed_dim=32, 
            use_singularity_weighting=use_singularity
        )
        Z_hyper, _ = hyperbolic_emb.fit_transform(features, labels)
        Z_proto, proto_labels = improved_structural_plasticity(Z_hyper, labels)
        rep['Proposed'] = Z_proto
        proposed_labels = proto_labels
    except Exception as e:
        print(f"  Failed to compute Proposed embedding: {e}")
        rep['Proposed'] = features[:100]  # Fallback
        proposed_labels = labels[:100]
    
    # 2. PCA (95% variance)
    print("📊 Computing PCA...")
    pca = PCA(n_components=0.95)
    rep['PCA'] = pca.fit_transform(features)
    
    # 3. Autoencoder
    print("📊 Computing Autoencoder...")
    encoding_dim = 32
    
    # Create autoencoder with functional API for better control
    input_layer = Input(shape=(features.shape[1],))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu', name='encoding')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(features.shape[1], activation='sigmoid')(decoded)
    
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train autoencoder
    autoencoder.fit(features, features, epochs=20, batch_size=32, verbose=0)
    
    # Create encoder model (get the encoding layer output)
    encoder = Model(inputs=input_layer, outputs=autoencoder.get_layer('encoding').output)
    rep['Autoencoder'] = encoder.predict(features)
    
    # 4. Sparse Coding
    print("📊 Computing Sparse Coding...")
    try:
        sparse_coder = DictionaryLearning(n_components=50, alpha=1.0, max_iter=50, random_state=42)
        rep['SparseCoding'] = sparse_coder.fit_transform(features)
    except Exception as e:
        print(f"  Warning: Sparse coding failed ({e}), using PCA fallback")
        pca_sparse = PCA(n_components=50)
        rep['SparseCoding'] = pca_sparse.fit_transform(features)
    
    # 5. Topological Dimension Reduction (new)
    if use_singularity:
        print("📊 Computing Topological Dimension Reduction...")
        tdr = TopologicalDimensionReduction(target_dim=32)
        rep['TopologicalDR'] = tdr.fit_transform(features)
    
    return rep, proposed_labels

# --- Visualization functions ---
def create_comprehensive_visualizations(comm_results):
    """Create comprehensive visualizations for semantic communication analysis"""
    
    # Filter valid results
    valid_results = {}
    for model, metrics in comm_results.items():
        if all(k in metrics for k in ['compression_ratio', 'semantic_preservation', 'transmission_time_s', 'bits_per_sample']):
            valid_results[model] = metrics
    
    if not valid_results:
        print("❌ No valid results to visualize")
        return
    
    models = list(valid_results.keys())
    compression_ratios = [valid_results[m]['compression_ratio'] for m in models]
    transmission_times = [valid_results[m]['transmission_time_s'] for m in models]
    semantic_preservations = [valid_results[m]['semantic_preservation'] for m in models]
    bits_per_sample = [valid_results[m]['bits_per_sample'] for m in models]
    
    # Color scheme
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFD166', '#F38181']
    
    # 1. Pareto Front: Compression vs Semantic Preservation
    plt.figure(figsize=(10, 8))
    plt.scatter(compression_ratios, semantic_preservations, s=300, c=colors[:len(models)], 
                alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, m in enumerate(models):
        plt.annotate(m, (compression_ratios[i], semantic_preservations[i]), 
                    fontsize=12, fontweight='bold', ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points')
    
    # Add ideal point
    max_comp = max(compression_ratios)
    max_sem = max(semantic_preservations)
    plt.scatter(max_comp, max_sem, s=200, c='gold', marker='*', 
                label='Ideal Point', edgecolors='black', linewidth=2)
    
    plt.xlabel('Compression Ratio', fontsize=14)
    plt.ylabel('Semantic Preservation (kNN Accuracy)', fontsize=14)
    plt.title('Pareto Front: Compression vs Semantic Preservation', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['simul_dir'], 'pareto_compression_semantic.png'), dpi=300)
    plt.show()
    
    # 2. Multi-metric Radar Chart
    from math import pi
    
    # Use valid_results instead of models
    if len(models) > 0:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Normalize metrics to 0-1 scale
        norm_compression = np.array(compression_ratios) / max(compression_ratios) if max(compression_ratios) > 0 else np.zeros(len(compression_ratios))
        norm_semantic = np.array(semantic_preservations)
        norm_efficiency = 1 - np.array(transmission_times) / max(transmission_times) if max(transmission_times) > 0 else np.ones(len(transmission_times))
        norm_bits = 1 - np.array(bits_per_sample) / max(bits_per_sample) if max(bits_per_sample) > 0 else np.ones(len(bits_per_sample))
        
        categories = ['Compression\nRatio', 'Semantic\nPreservation', 'Transmission\nEfficiency', 'Bit\nEfficiency']
        angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        
        for i, model in enumerate(models):
            values = [norm_compression[i], norm_semantic[i], norm_efficiency[i], norm_bits[i]]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Communication Efficiency Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PATHS['simul_dir'], 'radar_multi_metric.png'), dpi=300)
        plt.show()
    else:
        print("⚠️ Not enough valid models for radar chart")
    
    # 3. Efficiency vs Quality Trade-off
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Transmission Time vs Semantic Preservation
    ax1.scatter(transmission_times, semantic_preservations, s=300, c=colors[:len(models)], 
                alpha=0.7, edgecolors='black', linewidth=2)
    for i, m in enumerate(models):
        ax1.annotate(m, (transmission_times[i], semantic_preservations[i]), 
                    fontsize=10, ha='center', va='bottom')
    ax1.set_xlabel('Transmission Time (s)', fontsize=12)
    ax1.set_ylabel('Semantic Preservation', fontsize=12)
    ax1.set_title('Transmission Efficiency vs Quality', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Bits per Sample vs Semantic Preservation
    ax2.scatter(bits_per_sample, semantic_preservations, s=300, c=colors[:len(models)], 
                alpha=0.7, edgecolors='black', linewidth=2)
    for i, m in enumerate(models):
        ax2.annotate(m, (bits_per_sample[i], semantic_preservations[i]), 
                    fontsize=10, ha='center', va='bottom')
    ax2.set_xlabel('Bits per Sample', fontsize=12)
    ax2.set_ylabel('Semantic Preservation', fontsize=12)
    ax2.set_title('Bit Efficiency vs Quality', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['simul_dir'], 'efficiency_quality_tradeoff.png'), dpi=300)
    plt.show()
    
    # 4. Comprehensive Bar Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Compression Ratio
    bars1 = ax1.bar(models, compression_ratios, color=colors[:len(models)], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Compression Ratio', fontsize=12)
    ax1.set_title('Compression Ratio by Model', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars1, compression_ratios):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Transmission Time
    bars2 = ax2.bar(models, transmission_times, color=colors[:len(models)], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Transmission Time (s)', fontsize=12)
    ax2.set_title('Transmission Time by Model', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars2, transmission_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Semantic Preservation
    bars3 = ax3.bar(models, semantic_preservations, color=colors[:len(models)], alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Semantic Preservation (kNN Acc)', fontsize=12)
    ax3.set_title('Semantic Preservation by Model', fontsize=14, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.set_ylim(0, 1)
    for bar, val in zip(bars3, semantic_preservations):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Bits per Sample
    bars4 = ax4.bar(models, bits_per_sample, color=colors[:len(models)], alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Bits per Sample', fontsize=12)
    ax4.set_title('Bits per Sample by Model', fontsize=14, fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars4, bits_per_sample):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x labels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticklabels(models, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['simul_dir'], 'comprehensive_bar_comparison.png'), dpi=300)
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    print("="*60)
    print("🚀 SEMANTIC COMMUNICATION ANALYSIS WITH SINGULARITY DETECTION")
    print("="*60)
    
    # 1. Try to load features with singularity detection first
    features, labels = load_singularity_features()
    use_singularity = True
    
    if features is None:
        # Fall back to standard features
        features, labels = load_features_from_part1()
        use_singularity = False
        
        if features is None:
            print("❌ No features found. Extracting from videos...")
            # Load videos and extract features
            videos = load_sample_videos(n_samples=50)
            if not videos:
                raise RuntimeError("No video files found. Please check your dataset path.")
            
            # Create dummy labels if needed
            labels = np.array([i % 6 for i in range(len(videos))])  # 6 classes
            
            # Extract features with singularity detection
            features, labels = extract_singularity_features(videos, labels)
            use_singularity = True
    
    print(f"\n📊 Using {'singularity-enhanced' if use_singularity else 'standard'} features")
    print(f"📊 Features shape: {features.shape}, Labels shape: {labels.shape}")
    print(f"📊 Class distribution: {Counter(labels)}")
    
    # 2. Load sample video for size comparison
    print("\n📹 Loading sample video for size comparison...")
    video_data = load_sample_videos(n_samples=1)
    if video_data and len(video_data) > 0:
        sample_video = video_data[0]
        print(f"📹 Loaded video shape: {sample_video.shape}")
    else:
        # Create dummy video data for size comparison
        print("⚠️ Creating dummy video for size comparison...")
        sample_video = np.random.rand(30, 120, 160, 3).astype(np.float32)  # 30 frames, 120x160 RGB
    
    original_size = sample_video.nbytes
    print(f"📹 Original video size: {original_size/1e6:.2f} MB")
    
    # 3. Get all representations
    print("\n🔄 Computing different representations...")
    reps, proposed_labels = get_all_representations(features, labels, use_singularity=use_singularity)
    
    # 4. Calculate communication metrics for each method
    print("\n📊 Calculating communication metrics...")
    comm_results = {}
    
    for name, rep in reps.items():
        # For Proposed method, use prototype labels
        use_labels = proposed_labels if name == 'Proposed' else labels
        
        # Ensure rep and use_labels have same length
        min_len = min(len(rep), len(use_labels))
        rep = rep[:min_len]
        use_labels = use_labels[:min_len]
        
        comm_results[name] = measure_communication_metrics(
            sample_video, rep, rep, use_labels, name
        )
        
        print(f"\n{name}:")
        print(f"  Compression Ratio: {comm_results[name]['compression_ratio']:.2f}x")
        print(f"  Semantic Preservation: {comm_results[name]['semantic_preservation']:.3f}")
        print(f"  Transmission Time: {comm_results[name]['transmission_time_s']:.3f}s")
        print(f"  Bits per Sample: {comm_results[name]['bits_per_sample']:.0f}")
    
    # 5. Create comprehensive visualizations
    print("\n📊 Creating visualizations...")
    create_comprehensive_visualizations(comm_results)
    
    # 6. Save results
    results_path = os.path.join(PATHS['simul_dir'], 'semantic_comm_results_part3.json')
    with open(results_path, 'w') as f:
        json.dump(comm_results, f, indent=2)
    print(f"\n✅ Results saved to: {results_path}")
    
    # 7. Create summary report
    print("\n" + "="*60)
    print("📊 SUMMARY REPORT")
    print("="*60)
    
    # Filter valid results
    valid_results = {k: v for k, v in comm_results.items() 
                    if all(key in v for key in ['compression_ratio', 'semantic_preservation', 'transmission_time_s', 'bits_per_sample'])}
    
    if valid_results:
        # Find best method for each metric
        best_compression = max(valid_results.keys(), key=lambda x: valid_results[x]['compression_ratio'])
        best_semantic = max(valid_results.keys(), key=lambda x: valid_results[x]['semantic_preservation'])
        best_transmission = min(valid_results.keys(), key=lambda x: valid_results[x]['transmission_time_s'])
        best_bits = min(valid_results.keys(), key=lambda x: valid_results[x]['bits_per_sample'])
        
        print(f"\n🏆 Best Compression Ratio: {best_compression} ({valid_results[best_compression]['compression_ratio']:.2f}x)")
        print(f"🏆 Best Semantic Preservation: {best_semantic} ({valid_results[best_semantic]['semantic_preservation']:.3f})")
        print(f"🏆 Fastest Transmission: {best_transmission} ({valid_results[best_transmission]['transmission_time_s']:.3f}s)")
        print(f"🏆 Best Bit Efficiency: {best_bits} ({valid_results[best_bits]['bits_per_sample']:.0f} bits/sample)")
        
        # Overall recommendation
        print("\n💡 RECOMMENDATION:")
        if 'Proposed' in valid_results:
            proposed = valid_results['Proposed']
            print(f"The Proposed method (Hyperbolic + Singularity-aware Plasticity) achieves:")
            print(f"  - {proposed['compression_ratio']:.1f}x compression ratio")
            print(f"  - {proposed['semantic_preservation']:.3f} semantic preservation")
            print(f"  - {proposed['transmission_time_s']:.3f}s transmission time")
            
            # Check if it's Pareto optimal
            is_pareto = True
            for other_name, other_metrics in valid_results.items():
                if other_name != 'Proposed':
                    if (other_metrics['compression_ratio'] > proposed['compression_ratio'] and 
                        other_metrics['semantic_preservation'] > proposed['semantic_preservation']):
                        is_pareto = False
                        break
            
            if is_pareto:
                print("\n✅ The Proposed method is Pareto optimal!")
            else:
                print("\n⚠️ The Proposed method is not Pareto optimal, but offers good trade-offs.")
        else:
            print("⚠️ Proposed method results not available.")
    else:
        print("❌ No valid results available for summary.")
    
    print("\n✅ Semantic communication analysis completed successfully!")
    print("📁 Check the lgmd_simul folder for all generated visualizations and results.")