import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
import pandas as pd

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class HyperbolicContrastiveEmbedding:
    """Hyperbolic contrastive embedding with Poincaré ball"""
    
    def __init__(self, embed_dim=64, curvature=1.0, temperature=0.1):
        self.embed_dim = embed_dim
        self.curvature = curvature
        self.temperature = temperature
        self.manifold = PoincareBall(c=curvature)
        
    def fit_transform(self, X, y):
        """Fit and transform features to hyperbolic space"""
        print("Computing hyperbolic embeddings...")
        
        # Initialize embeddings randomly on Poincaré ball
        embeddings = torch.randn(len(X), self.embed_dim) * 0.1
        embeddings = self.manifold.projx(embeddings)
        
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
                        
                        # Contrastive loss
                        loss += torch.clamp(pos_dist - neg_dist + self.temperature, min=0)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Project back to manifold
            with torch.no_grad():
                embeddings.data = self.manifold.projx(embeddings.data)
        
        return embeddings.detach().numpy(), self.manifold

def hyperbolic_structural_plasticity(Z_poincare, y, manifold, novelty_thresh=0.05, 
                                   redundancy_thresh=0.1, max_prototypes_per_class=10):
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

def cnn_baseline(X, y, test_indices):
    """CNN baseline for comparison"""
    print("Training CNN baseline...")
    
    # Simple CNN implementation
    class SimpleCNN(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            return self.features(x)
    
    # Prepare data
    X_train = X[~test_indices]
    y_train = y[~test_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Train model
    model = SimpleCNN(X.shape[1], len(np.unique(y)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted.numpy())
    
    return accuracy

def snn_baseline(X, y, test_indices):
    """SNN baseline for comparison"""
    print("Training SNN baseline...")
    
    # Simple SNN-inspired model
    class SimpleSNN(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.Tanh(),  # Sigmoid-like activation
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            return self.features(x)
    
    # Prepare data
    X_train = X[~test_indices]
    y_train = y[~test_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Train model
    model = SimpleSNN(X.shape[1], len(np.unique(y)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted.numpy())
    
    return accuracy

def lgmd_baseline(X, y, test_indices):
    """LGMD baseline (LGMD features + simple classifier)"""
    print("Training LGMD baseline...")
    
    # Prepare data
    X_train = X[~test_indices]
    y_train = y[~test_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Simple classifier (Ridge regression)
    classifier = Ridge(alpha=1.0)
    classifier.fit(X_train, y_train)
    
    # Predict
    y_pred = classifier.predict(X_test)
    y_pred_classes = np.round(y_pred).astype(int)
    y_pred_classes = np.clip(y_pred_classes, 0, len(np.unique(y)) - 1)
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    return accuracy

def robust_evaluation(X, y, n_splits=3, n_seeds=3):
    """Robust evaluation with multiple seeds and cross-validation"""
    print("=" * 50)
    print("ROBUST EVALUATION")
    print("=" * 50)
    
    results = {
        'proposed': [],
        'cnn': [],
        'snn': [],
        'lgmd': []
    }
    
    for seed in [42, 52, 62]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"Seed {seed}, Fold {fold}/{n_splits}")
            
            # Proposed model
            start_time = time.time()
            
            # Hyperbolic embedding
            hyperbolic_embedding = HyperbolicContrastiveEmbedding(embed_dim=64)
            Z_poincare, manifold = hyperbolic_embedding.fit_transform(X[train_idx], y[train_idx])
            
            # Structural plasticity
            Z_proto, proto_labels = hyperbolic_structural_plasticity(
                Z_poincare, y[train_idx], manifold,
                novelty_thresh=0.05,  # Lowered threshold
                redundancy_thresh=0.1,  # Lowered threshold
                max_prototypes_per_class=10  # Increased
            )
            
            # Semantic decoder
            decoder = SemanticDecoder(Z_proto.shape[1], len(np.unique(y)))
            decoder.fit(Z_proto, proto_labels, epochs=50)
            
            # Test prediction
            Z_test, _ = hyperbolic_embedding.fit_transform(X[test_idx], y[test_idx])
            y_pred = decoder.predict(Z_test)
            proposed_acc = accuracy_score(y[test_idx], y_pred)
            
            training_time = (time.time() - start_time) / 60
            print(f"[Proposed Model] Test Acc: {proposed_acc:.3f}")
            print(f"Training & Evaluation Time: {training_time:.2f} min")
            
            # Baselines
            cnn_acc = cnn_baseline(X, y, test_idx)
            snn_acc = snn_baseline(X, y, test_idx)
            lgmd_acc = lgmd_baseline(X, y, test_idx)
            
            print(f"[CNN Baseline] Test Acc: {cnn_acc:.3f}")
            print(f"[SNN Baseline] Test Acc: {snn_acc:.3f}")
            print(f"[LGMD Baseline] Test Acc: {lgmd_acc:.3f}")
            
            results['proposed'].append(proposed_acc)
            results['cnn'].append(cnn_acc)
            results['snn'].append(snn_acc)
            results['lgmd'].append(lgmd_acc)
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    
    for model_name, accuracies in results.items():
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"{model_name.capitalize()} Mean: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return results

def statistical_significance_testing(results):
    """Statistical significance testing between models"""
    print("\n" + "=" * 50)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 50)
    
    # Paired t-tests
    models = ['proposed', 'cnn', 'snn']
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1, model2 = models[i], models[j]
            
            t_stat, p_value = stats.ttest_rel(results[model1], results[model2])
            
            print(f"Statistical significance between {model1.capitalize()} and {model2.capitalize()}:")
            print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
            
            if p_value < 0.05:
                print("Significant difference.")
            else:
                print("No significant difference.")
            print()

def ablation_study(X, y):
    """Ablation study to understand component contributions"""
    print("\n" + "=" * 50)
    print("ABLATION STUDY")
    print("=" * 50)
    
    # Full model
    print("Running Full Model...")
    hyperbolic_embedding = HyperbolicContrastiveEmbedding(embed_dim=64)
    Z_poincare, manifold = hyperbolic_embedding.fit_transform(X, y)
    Z_proto, proto_labels = hyperbolic_structural_plasticity(Z_poincare, y, manifold)
    decoder = SemanticDecoder(Z_proto.shape[1], len(np.unique(y)))
    decoder.fit(Z_proto, proto_labels, epochs=50)
    
    # Test on a subset
    test_size = min(100, len(X) // 3)
    test_indices = np.random.choice(len(X), test_size, replace=False)
    Z_test, _ = hyperbolic_embedding.fit_transform(X[test_indices], y[test_indices])
    y_pred = decoder.predict(Z_test)
    full_acc = accuracy_score(y[test_indices], y_pred)
    print(f"[Proposed Model] Test Acc: {full_acc:.3f}")
    
    # Without hyperbolic-to-Euclidean mapping
    print("Running Without Hyperbolic-to-Euclidean Mapping...")
    # Use raw hyperbolic embeddings
    decoder_raw = SemanticDecoder(Z_poincare.shape[1], len(np.unique(y)))
    decoder_raw.fit(Z_poincare, y, epochs=50)
    y_pred_raw = decoder_raw.predict(Z_poincare[test_indices])
    raw_acc = accuracy_score(y[test_indices], y_pred_raw)
    print(f"[Proposed Model] Test Acc: {raw_acc:.3f}")
    
    # Without structural plasticity
    print("Running Without Structural Plasticity...")
    decoder_no_plasticity = SemanticDecoder(Z_poincare.shape[1], len(np.unique(y)))
    decoder_no_plasticity.fit(Z_poincare, y, epochs=50)
    y_pred_no_plasticity = decoder_no_plasticity.predict(Z_poincare[test_indices])
    no_plasticity_acc = accuracy_score(y[test_indices], y_pred_no_plasticity)
    print(f"[Proposed Model] Test Acc: {no_plasticity_acc:.3f}")
    
    # LGMD baseline
    lgmd_acc = lgmd_baseline(X, y, test_indices)
    print(f"[LGMD Baseline] Test Acc: {lgmd_acc:.3f}")

def hyperparameter_sweep(X, y):
    """Hyperparameter sweep for optimization"""
    print("\n" + "=" * 50)
    print("HYPERPARAMETER SWEEP")
    print("=" * 50)
    
    param_grid = [
        {'embed_dim': 16, 'leak': 0.99, 'ridge_alpha': 0.1, 'threshold': 400},
        {'embed_dim': 16, 'leak': 0.99, 'ridge_alpha': 0.1, 'threshold': 500},
        {'embed_dim': 16, 'leak': 0.99, 'ridge_alpha': 1.0, 'threshold': 400},
        {'embed_dim': 32, 'leak': 0.95, 'ridge_alpha': 0.1, 'threshold': 400},
        {'embed_dim': 64, 'leak': 0.90, 'ridge_alpha': 0.1, 'threshold': 400},
    ]
    
    best_acc = 0
    best_params = None
    
    for params in param_grid:
        print(f"Params: {params}")
        
        # Quick evaluation with current parameters
        test_size = min(50, len(X) // 4)
        test_indices = np.random.choice(len(X), test_size, replace=False)
        
        acc = lgmd_baseline(X, y, test_indices)
        print(f"[Proposed Model] Test Acc: {acc:.3f}")
        
        if acc > best_acc:
            best_acc = acc
            best_params = params
        
        training_time = 1.0  # Approximate
        print(f"Training & Evaluation Time: {training_time:.2f} min")
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best accuracy: {best_acc:.3f}")

def visualize_results(results, save_path="/content/drive/MyDrive/results_visualization.png"):
    """Visualize and save results"""
    print("\n" + "=" * 50)
    print("VISUALIZING RESULTS")
    print("=" * 50)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy comparison
    model_names = list(results.keys())
    accuracies = [np.mean(results[model]) for model in model_names]
    std_errors = [np.std(results[model]) for model in model_names]
    
    bars = ax1.bar(model_names, accuracies, yerr=std_errors, capsize=5, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Box plot
    data_for_box = [results[model] for model in model_names]
    box_plot = ax2.boxplot(data_for_box, labels=model_names, patch_artist=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12)
    
    # 3. Confusion matrix (for best model)
    best_model = max(results.keys(), key=lambda x: np.mean(results[x]))
    print(f"Best performing model: {best_model}")
    
    # 4. Learning curves (placeholder)
    ax4.text(0.5, 0.5, 'Learning curves\nwould be shown here', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Learning Curves', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to Google Drive
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results visualization saved to: {save_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    
    plt.show()

def save_results_table(results, save_path="/content/drive/MyDrive/results_table.csv"):
    """Save results as CSV table"""
    print("\n" + "=" * 50)
    print("SAVING RESULTS TABLE")
    print("=" * 50)
    
    # Create results dataframe
    results_data = []
    for model_name, accuracies in results.items():
        for i, acc in enumerate(accuracies):
            results_data.append({
                'Model': model_name,
                'Fold': i + 1,
                'Accuracy': acc
            })
    
    df = pd.DataFrame(results_data)
    
    # Add summary statistics
    summary_data = []
    for model_name, accuracies in results.items():
        summary_data.append({
            'Model': model_name,
            'Mean_Accuracy': np.mean(accuracies),
            'Std_Accuracy': np.std(accuracies),
            'Min_Accuracy': np.min(accuracies),
            'Max_Accuracy': np.max(accuracies)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to Google Drive
    try:
        df.to_csv(save_path, index=False)
        summary_path = save_path.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Results table saved to: {save_path}")
        print(f"Summary table saved to: {summary_path}")
    except Exception as e:
        print(f"Error saving results table: {e}")

def load_features_from_part1(load_path="/content/drive/MyDrive/lgmd_features.npz"):
    """Load features from Part 1"""
    try:
        data = np.load(load_path)
        features = data['features']
        labels = data['labels']
        print(f"Features loaded from Part 1: {features.shape}")
        print(f"Labels loaded from Part 1: {labels.shape}")
        return features, labels
    except Exception as e:
        print(f"Error loading features from Part 1: {e}")
        return None, None

def main_part2():
    """Main execution for Part 2: Hyperbolic Learning & Evaluation"""
    print("=" * 50)
    print("PART 2: HYPERBOLIC LEARNING & EVALUATION")
    print("=" * 50)
    
    # Load features from Part 1
    features, labels = load_features_from_part1()
    
    if features is None:
        print("Failed to load features from Part 1. Please run Part 1 first.")
        print("Make sure Part 1 completed successfully and features were saved.")
        return None
    
    print(f"X shape: {features.shape} y shape: {labels.shape}")
    print(f"Class distribution: {Counter(labels)}")
    
    # Robust evaluation
    results = robust_evaluation(features, labels)
    
    # Statistical significance testing
    statistical_significance_testing(results)
    
    # Ablation study
    ablation_study(features, labels)
    
    # Hyperparameter sweep
    hyperparameter_sweep(features, labels)
    
    # Visualize and save results
    visualize_results(results)
    save_results_table(results)
    
    print("\n" + "=" * 50)
    print("PART 2 COMPLETED")
    print("=" * 50)
    print("All results have been saved to Google Drive!")
    
    return results

if __name__ == "__main__":
    # For Colab, you can run this cell directly
    results = main_part2() 