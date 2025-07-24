import os
import sys
sys.path.append('/content/drive/MyDrive')  # Ensure custom modules in Drive are importable
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf

# --- 경로 설정 ---
PATHS = {
    'drive_path': "/content/drive/MyDrive/KTH_dataset",
    'features_save_path': "/content/drive/MyDrive/lgmd_features.npz",
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

# --- 데이터 로딩 함수 ---
def load_features_from_part1(load_path=PATHS['features_save_path']):
    data = np.load(load_path)
    features = data['features']
    labels = data['labels']
    print(f"✅ Features loaded: {features.shape}, Labels: {labels.shape}")
    return features, labels

def load_sample_videos(n_samples=100, ext='avi'):
    import glob
    import cv2
    video_dir = PATHS['drive_path']
    video_files = glob.glob(os.path.join(video_dir, '*', f'*.{ext}'))
    print(f"Found {len(video_files)} video files with extension .{ext}")
    videos = []
    for vf in video_files[:n_samples]:
        try:
            cap = cv2.VideoCapture(vf)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            if frames:
                video_arr = np.stack(frames)
                videos.append(video_arr)
        except Exception as e:
            print(f"Failed to load {vf}: {e}")
    print(f"Loaded {len(videos)} videos from {video_dir}")
    return videos

# --- Proposed (Hyperbolic+Plasticity) ---
def get_proposed_embedding(features, labels, embed_dim=32):
    try:
        from improved_lgmd_hyperbolic_pipeline import HyperbolicContrastiveEmbedding, hyperbolic_structural_plasticity
    except ImportError as e:
        print("❌ ERROR: improved_lgmd_hyperbolic_pipeline.py not found or import failed. Please check the file in /content/drive/MyDrive.")
        raise e
    emb = HyperbolicContrastiveEmbedding(embed_dim=embed_dim)
    Z, manifold = emb.fit_transform(features, labels)
    Z_proto, proto_labels = hyperbolic_structural_plasticity(Z, labels, manifold)
    return Z_proto, proto_labels

# --- Semantic Preservation (kNN Consistency) ---
def evaluate_semantic_preservation(features, labels):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, features, labels, cv=3)
    return np.mean(scores)

# --- Compression/Transmission/Preservation Metrics ---
def measure_communication_metrics(original_data, compressed_data, features, labels, name):
    original_size = original_data.nbytes
    compressed_size = compressed_data.nbytes
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    bandwidth = 1e6  # 1 Mbps
    transmission_time = (compressed_size * 8) / bandwidth
    semantic_preservation = evaluate_semantic_preservation(features, labels)
    return {
        'compression_ratio': compression_ratio,
        'original_size_MB': original_size / 1e6,
        'compressed_size_MB': compressed_size / 1e6,
        'transmission_time_s': transmission_time,
        'semantic_preservation': semantic_preservation
    }

# --- Baseline/Proposed Feature Extraction ---
def get_all_representations(features, labels):
    from sklearn.decomposition import PCA, DictionaryLearning
    from tensorflow.keras import Input, Sequential
    from tensorflow.keras.layers import Dense
    rep = {}
    # 1. Proposed (Hyperbolic+Plasticity)
    try:
        rep['Proposed'], proposed_labels = get_proposed_embedding(features, labels)
    except Exception as e:
        print(f"Failed to compute Proposed embedding: {e}")
        rep['Proposed'] = features
        proposed_labels = labels
    # 2. PCA (95% variance)
    pca = PCA(n_components=0.95)
    rep['PCA'] = pca.fit_transform(features)
    # 3. Autoencoder (with explicit Input layer)
    encoding_dim = 32
    autoencoder = Sequential([
        Input(shape=(features.shape[1],)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(encoding_dim, activation='relu', name='encoding'),
        Dense(128, activation='relu'),
        Dense(256, activation='relu'),
        Dense(features.shape[1], activation='sigmoid')
    ])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(features, features, epochs=30, batch_size=32, verbose=0, validation_split=0.1)
    encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoding').output)
    rep['Autoencoder'] = encoder.predict(features)
    # 4. Sparse Coding
    sparse_coder = DictionaryLearning(n_components=50, alpha=1.0, max_iter=100, random_state=42)
    rep['SparseCoding'] = sparse_coder.fit_transform(features)
    return rep, proposed_labels

# --- 메인 실행 블록 ---
if __name__ == '__main__':
    features, labels = load_features_from_part1()
    video_data = load_sample_videos(n_samples=100, ext='avi')
    if not video_data:
        raise RuntimeError("No video files found in the dataset path. Please check your data and path.")
    sample_video = video_data[0]
    original_size = sample_video.nbytes

    # 1. 모든 알고리즘의 representation 추출
    reps, proposed_labels = get_all_representations(features, labels)

    # 2. 각 알고리즘별 semantic/communication metric 계산
    comm_results = {}
    for name, rep in reps.items():
        # Proposed는 prototype label 사용, 나머지는 원래 label 사용
        use_labels = proposed_labels if name == 'Proposed' else labels
        comm_results[name] = measure_communication_metrics(sample_video, rep, rep, use_labels, name)
        print(f"{name}: {comm_results[name]}")

    # 3. 다양한 시각화
    models = list(comm_results.keys())
    compression_ratios = [comm_results[m]['compression_ratio'] for m in models]
    transmission_times = [comm_results[m]['transmission_time_s'] for m in models]
    semantic_preservations = [comm_results[m]['semantic_preservation'] for m in models]

    # 3-1. Compression vs Semantic Preservation (Pareto)
    plt.figure(figsize=(8,6))
    plt.scatter(compression_ratios, semantic_preservations, s=200, c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFD166'])
    for i, m in enumerate(models):
        plt.annotate(m, (compression_ratios[i], semantic_preservations[i]), fontsize=12, fontweight='bold')
    plt.xlabel('Compression Ratio')
    plt.ylabel('Semantic Preservation (kNN Acc)')
    plt.title('Pareto: Compression vs Semantic Preservation')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PATHS['simul_dir'], 'pareto_compression_semantic.png'), dpi=300)
    plt.show()

    # 3-2. Transmission Time Bar
    plt.figure(figsize=(7,5))
    plt.bar(models, transmission_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFD166'])
    plt.ylabel('Transmission Time (s)')
    plt.title('Transmission Time by Model')
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(PATHS['simul_dir'], 'transmission_time.png'), dpi=300)
    plt.show()

    # 3-3. Compression Ratio Bar
    plt.figure(figsize=(7,5))
    plt.bar(models, compression_ratios, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFD166'])
    plt.ylabel('Compression Ratio')
    plt.title('Compression Ratio by Model')
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(PATHS['simul_dir'], 'compression_ratio.png'), dpi=300)
    plt.show()

    # 3-4. Semantic Preservation Bar
    plt.figure(figsize=(7,5))
    plt.bar(models, semantic_preservations, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFD166'])
    plt.ylabel('Semantic Preservation (kNN Acc)')
    plt.title('Semantic Preservation by Model')
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(PATHS['simul_dir'], 'semantic_preservation.png'), dpi=300)
    plt.show()

    # 4. 결과 저장
    import json
    with open(os.path.join(PATHS['simul_dir'], 'semantic_comm_results_part3.json'), 'w') as f:
        json.dump(comm_results, f, indent=2)
    print(f"✅ Results saved to: {os.path.join(PATHS['simul_dir'], 'semantic_comm_results_part3.json')}") 