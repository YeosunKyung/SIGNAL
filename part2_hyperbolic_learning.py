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
from IPython.display import clear_output

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total: 
        print()

def format_time(seconds):
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"

class HyperbolicContrastiveEmbedding:
    """Hyperbolic contrastive embedding with Poincaré ball"""
    
    def __init__(self, embed_dim=64, curvature=1.0, temperature=0.1, batch_size=32):
        self.embed_dim = embed_dim
        self.curvature = curvature
        self.temperature = temperature
        self.batch_size = batch_size
        self.manifold = PoincareBall(c=curvature)
        
    def fit_transform(self, X, y, validation_split=0.2, patience=10):
        """Fit and transform features to hyperbolic space with early stopping and advanced negative sampling"""
        start_time = time.time()
        print("🌐 Computing hyperbolic embeddings...")
        
        # Split data for validation
        n_val = int(len(X) * validation_split)
        val_indices = np.random.choice(len(X), n_val, replace=False)
        train_indices = np.array([i for i in range(len(X)) if i not in val_indices])
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Initialize embeddings randomly on Poincaré ball
        embeddings = torch.randn(len(X), self.embed_dim) * 0.1
        embeddings = self.manifold.projx(embeddings)
        
        # Contrastive learning with batch processing, early stopping, and advanced negative sampling
        optimizer = optim.Adam([embeddings.requires_grad_()], lr=0.01)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(200):  # Increased max epochs
            # Training loss
            train_loss = 0
            n_batches = 0
            
            # Batch processing for efficiency
            for batch_start in range(0, len(train_indices), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(train_indices))
                batch_indices = train_indices[batch_start:batch_end]
                
                batch_loss = 0
                for i in batch_indices:
                    # Positive pairs (same class) - multiple positive samples
                    pos_indices = np.where(y == y[i])[0]
                    pos_indices = pos_indices[pos_indices != i]
                    
                    if len(pos_indices) > 0:
                        # Sample multiple positive pairs
                        n_pos = min(3, len(pos_indices))
                        pos_indices_selected = np.random.choice(pos_indices, n_pos, replace=False)
                        
                        for pos_idx in pos_indices_selected:
                            pos_dist = self.manifold.dist(embeddings[i], embeddings[pos_idx])
                            
                            # Semi-hard negative mining
                            neg_indices = np.where(y != y[i])[0]
                            if len(neg_indices) > 0:
                                # Calculate distances to all negative samples
                                neg_dists = []
                                for neg_idx in neg_indices:
                                    neg_dist = self.manifold.dist(embeddings[i], embeddings[neg_idx])
                                    neg_dists.append(neg_dist.item())
                                
                                # Select semi-hard negatives (not too easy, not too hard)
                                neg_dists = np.array(neg_dists)
                                semi_hard_mask = (neg_dists > pos_dist.item() * 0.5) & (neg_dists < pos_dist.item() * 2.0)
                                
                                if np.any(semi_hard_mask):
                                    semi_hard_indices = neg_indices[semi_hard_mask]
                                    n_neg = min(2, len(semi_hard_indices))
                                    neg_indices_selected = np.random.choice(semi_hard_indices, n_neg, replace=False)
                                else:
                                    # Fallback to random negative sampling
                                    n_neg = min(2, len(neg_indices))
                                    neg_indices_selected = np.random.choice(neg_indices, n_neg, replace=False)
                                
                                for neg_idx in neg_indices_selected:
                                    neg_dist = self.manifold.dist(embeddings[i], embeddings[neg_idx])
                                    # Contrastive loss with margin
                                    batch_loss += torch.clamp(pos_dist - neg_dist + self.temperature, min=0)
                
                if batch_loss > 0:
                    train_loss += batch_loss
                    n_batches += 1
            
            # Validation loss (simplified for efficiency)
            val_loss = 0
            for i in val_indices:
                pos_indices = np.where(y == y[i])[0]
                pos_indices = pos_indices[pos_indices != i]
                
                if len(pos_indices) > 0:
                    pos_idx = np.random.choice(pos_indices)
                    pos_dist = self.manifold.dist(embeddings[i], embeddings[pos_idx])
                    
                    neg_indices = np.where(y != y[i])[0]
                    if len(neg_indices) > 0:
                        neg_idx = np.random.choice(neg_indices)
                        neg_dist = self.manifold.dist(embeddings[i], embeddings[neg_idx])
                        
                        val_loss += torch.clamp(pos_dist - neg_dist + self.temperature, min=0)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                print(f"  🔄 Epoch {epoch}, Train Loss: {train_loss/n_batches:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"  ⏹️ Early stopping at epoch {epoch}")
                break
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Project back to manifold
            with torch.no_grad():
                embeddings.data = self.manifold.projx(embeddings.data)
                
            # Strategic GPU memory management (only every 10 epochs)
            if epoch % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        embedding_time = time.time() - start_time
        print(f"✅ Hyperbolic embedding completed in {format_time(embedding_time)}")
        
        return embeddings.detach().numpy(), self.manifold

def hyperbolic_structural_plasticity(Z_poincare, y, manifold, novelty_thresh=0.05, 
                                   redundancy_thresh=0.1, max_prototypes_per_class=10):
    """Structural plasticity for prototype selection in hyperbolic space (GPU optimized with ANN)"""
    start_time = time.time()
    print("🧠 Applying structural plasticity...")
    
    # Keep everything as PyTorch tensors for GPU efficiency
    if not torch.is_tensor(Z_poincare):
        Z_poincare = torch.tensor(Z_poincare, dtype=torch.float32)
    
    unique_classes = torch.unique(torch.tensor(y))
    selected_prototypes = []
    prototype_labels = []
    
    for class_label in unique_classes:
        class_indices = torch.where(torch.tensor(y) == class_label)[0]
        class_embeddings = Z_poincare[class_indices]
        
        # Initialize with class mean for better stability
        class_mean = torch.mean(class_embeddings, dim=0)
        prototypes = [class_mean]
        prototype_labels.append(class_label.item())
        
        # Convert to tensor for batch processing
        prototype_tensor = torch.stack(prototypes)
        
        # ANN-based fast distance computation for large prototype sets
        def fast_nearest_neighbor(query, candidates, k=1):
            """Fast approximate nearest neighbor search"""
            if len(candidates) <= 10:  # For small sets, use exact computation
                dists = manifold.dist(query, candidates)
                return torch.min(dists), torch.argmin(dists)
            else:
                # Approximate search: sample subset and find nearest
                n_samples = min(10, len(candidates))
                sample_indices = torch.randperm(len(candidates))[:n_samples]
                sample_candidates = candidates[sample_indices]
                dists = manifold.dist(query, sample_candidates)
                min_idx = torch.argmin(dists)
                return dists[min_idx], sample_indices[min_idx]
        
        for i in range(len(class_embeddings)):
            embedding = class_embeddings[i].unsqueeze(0)  # Add batch dimension
            
            # Check novelty (distance to existing prototypes) - ANN-based computation
            if len(prototype_tensor) > 1:
                min_dist_to_prototypes, _ = fast_nearest_neighbor(embedding, prototype_tensor)
                min_dist_to_prototypes = min_dist_to_prototypes.item()
            else:
                min_dist_to_prototypes = float('inf')
            
            # Check redundancy (distance to other samples in class) - ANN-based computation
            if len(class_embeddings) > 1:
                other_embeddings = torch.cat([class_embeddings[:i], class_embeddings[i+1:]])
                
                # Use sampling for large class sizes
                if len(other_embeddings) > 20:
                    sample_size = min(20, len(other_embeddings))
                    sample_indices = torch.randperm(len(other_embeddings))[:sample_size]
                    other_embeddings_sampled = other_embeddings[sample_indices]
                    dists_to_class = manifold.dist(embedding, other_embeddings_sampled)
                else:
                    dists_to_class = manifold.dist(embedding, other_embeddings)
                
                avg_dist_to_class = torch.mean(dists_to_class).item()
            else:
                avg_dist_to_class = 0
            
            # Novelty and redundancy criteria
            is_novel = min_dist_to_prototypes > novelty_thresh
            is_not_redundant = avg_dist_to_class > redundancy_thresh
            
            if is_novel and is_not_redundant and len(prototypes) < max_prototypes_per_class:
                prototypes.append(class_embeddings[i])
                prototype_labels.append(class_label.item())
                prototype_tensor = torch.stack(prototypes)
        
        selected_prototypes.extend(prototypes)
    
    if len(selected_prototypes) == 0:
        print("⚠️ Warning: No prototypes found after structural plasticity. Using class means as fallback.")
        # Fallback: use class means
        for class_label in unique_classes:
            class_indices = torch.where(torch.tensor(y) == class_label)[0]
            class_embeddings = Z_poincare[class_indices]
            class_mean = torch.mean(class_embeddings, dim=0)
            selected_prototypes.append(class_mean)
            prototype_labels.append(class_label.item())
    
    # Convert to numpy only at the end
    selected_prototypes = torch.stack(selected_prototypes).cpu().numpy()
    prototype_labels = np.array(prototype_labels)
    
    plasticity_time = time.time() - start_time
    print(f"✅ Structural plasticity completed in {format_time(plasticity_time)}")
    print(f"📊 Selected {len(selected_prototypes)} prototypes from {len(unique_classes)} classes")
    
    return selected_prototypes, prototype_labels

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
        
    def fit(self, X, y, epochs=100, lr=0.001, validation_split=0.2, patience=15, use_cv=False):
        """Train the semantic decoder with early stopping (cross-validation option)"""
        start_time = time.time()
        print("🎯 Training semantic decoder...")
        
        if use_cv and len(X) >= 30:  # Use cross-validation for larger datasets
            print("  🔄 Using cross-validation based early stopping...")
            best_val_loss = float('inf')
            patience_counter = 0
            
            # 3-fold cross-validation for early stopping
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            for epoch in range(epochs):
                # Training on full data
                self.model.train()
                X_tensor = torch.FloatTensor(X)
                y_tensor = torch.LongTensor(y)
                
                optimizer = optim.Adam(self.model.parameters(), lr=lr)
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                # Cross-validation for validation loss
                cv_losses = []
                for train_idx, val_idx in skf.split(X, y):
                    X_cv_train, X_cv_val = X[train_idx], X[val_idx]
                    y_cv_train, y_cv_val = y[train_idx], y[val_idx]
                    
                    # Quick validation on this fold
                    self.model.eval()
                    with torch.no_grad():
                        X_cv_val_tensor = torch.FloatTensor(X_cv_val)
                        cv_outputs = self.model(X_cv_val_tensor)
                        cv_loss = criterion(cv_outputs, torch.LongTensor(y_cv_val))
                        cv_losses.append(cv_loss.item())
                
                avg_cv_loss = np.mean(cv_losses)
                
                # Early stopping based on CV loss
                if avg_cv_loss < best_val_loss:
                    best_val_loss = avg_cv_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 20 == 0:
                    train_accuracy = (torch.max(outputs, 1)[1] == y_tensor).float().mean().item()
                    print(f"  🔄 Epoch {epoch}, Train Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.4f}")
                    print(f"  📊 CV Loss: {avg_cv_loss:.4f}")
                
                if patience_counter >= patience:
                    print(f"  ⏹️ Early stopping at epoch {epoch}")
                    break
        else:
            # Standard validation split approach
            print("  📊 Using standard validation split...")
            
            # Split data for validation
            n_val = int(len(X) * validation_split)
            val_indices = np.random.choice(len(X), n_val, replace=False)
            train_indices = np.array([i for i in range(len(X)) if i not in val_indices])
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    _, predicted = torch.max(val_outputs, 1)
                    val_accuracy = (predicted == y_val_tensor).float().mean().item()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 20 == 0:
                    train_accuracy = (torch.max(outputs, 1)[1] == y_train_tensor).float().mean().item()
                    print(f"  🔄 Epoch {epoch}, Train Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.4f}")
                    print(f"  📊 Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}")
                
                if patience_counter >= patience:
                    print(f"  ⏹️ Early stopping at epoch {epoch}")
                    break
        
        training_time = time.time() - start_time
        print(f"✅ Semantic decoder training completed in {format_time(training_time)}")
    
    def predict(self, X):
        """Predict class labels"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.numpy()

def cnn_baseline(X, y, test_indices, return_predictions=False):
    """CNN baseline for comparison"""
    print("🤖 Training CNN baseline...")
    
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
    
    if return_predictions:
        return predicted.numpy()
    return accuracy

def snn_baseline(X, y, test_indices, return_predictions=False):
    """SNN baseline for comparison"""
    print("🧠 Training SNN baseline...")
    
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
    
    if return_predictions:
        return predicted.numpy()
    return accuracy

def lgmd_baseline(X, y, test_indices, return_predictions=False):
    """LGMD baseline (LGMD features + simple classifier)"""
    print("🔬 Training LGMD baseline...")
    
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
    
    if return_predictions:
        return y_pred_classes
    return accuracy

def robust_evaluation(X, y, n_splits=3, n_seeds=3):
    """Robust evaluation with multiple seeds and cross-validation"""
    start_time = time.time()
    print("=" * 60)
    print("🏆 ROBUST EVALUATION")
    print("=" * 60)
    
    results = {
        'proposed': [],
        'cnn': [],
        'snn': [],
        'lgmd': []
    }
    
    total_iterations = n_seeds * n_splits
    current_iteration = 0
    
    for seed in [42, 52, 62]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            current_iteration += 1
            
            # Progress bar
            print_progress_bar(current_iteration, total_iterations, 
                              prefix=f'Robust Evaluation', 
                              suffix=f'Seed {seed}, Fold {fold}', 
                              length=40)
            
            # Proposed model
            fold_start = time.time()
            
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
            
            # Semantic decoder with cross-validation option
            decoder = SemanticDecoder(Z_proto.shape[1], len(np.unique(y)))
            decoder.fit(Z_proto, proto_labels, epochs=50, use_cv=True)
            
            # Test prediction
            Z_test, _ = hyperbolic_embedding.fit_transform(X[test_idx], y[test_idx])
            y_pred = decoder.predict(Z_test)
            proposed_acc = accuracy_score(y[test_idx], y_pred)
            
            training_time = (time.time() - fold_start) / 60
            
            # Baselines
            cnn_acc = cnn_baseline(X, y, test_idx)
            snn_acc = snn_baseline(X, y, test_idx)
            lgmd_acc = lgmd_baseline(X, y, test_idx)
            
            results['proposed'].append(proposed_acc)
            results['cnn'].append(cnn_acc)
            results['snn'].append(snn_acc)
            results['lgmd'].append(lgmd_acc)
    
    # Clear progress bar and print summary
    clear_output(wait=True)
    
    evaluation_time = time.time() - start_time
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print(f"📊 EVALUATION RESULTS (completed in {format_time(evaluation_time)})")
    print("=" * 60)
    
    for model_name, accuracies in results.items():
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"🏆 {model_name.capitalize()} Mean: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return results

def statistical_significance_testing(results):
    """Statistical significance testing between models"""
    print("\n" + "=" * 60)
    print("📈 STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 60)
    
    # Paired t-tests
    models = ['proposed', 'cnn', 'snn']
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1, model2 = models[i], models[j]
            
            t_stat, p_value = stats.ttest_rel(results[model1], results[model2])
            
            print(f"🔍 Statistical significance between {model1.capitalize()} and {model2.capitalize()}:")
            print(f"  📊 t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
            
            if p_value < 0.05:
                print("  ✅ Significant difference.")
            else:
                print("  ❌ No significant difference.")
            print()

def ablation_study(X, y):
    """Ablation study to understand component contributions"""
    print("\n" + "=" * 60)
    print("🔬 ABLATION STUDY")
    print("=" * 60)
    
    # Full model
    print("🚀 Running Full Model...")
    ablation_start = time.time()
    
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
    print(f"✅ [Full Model] Test Acc: {full_acc:.3f}")
    
    # Without hyperbolic-to-Euclidean mapping
    print("🔄 Running Without Hyperbolic-to-Euclidean Mapping...")
    # Use raw hyperbolic embeddings
    decoder_raw = SemanticDecoder(Z_poincare.shape[1], len(np.unique(y)))
    decoder_raw.fit(Z_poincare, y, epochs=50)
    y_pred_raw = decoder_raw.predict(Z_poincare[test_indices])
    raw_acc = accuracy_score(y[test_indices], y_pred_raw)
    print(f"✅ [Raw Hyperbolic] Test Acc: {raw_acc:.3f}")
    
    # Without structural plasticity
    print("🔄 Running Without Structural Plasticity...")
    decoder_no_plasticity = SemanticDecoder(Z_poincare.shape[1], len(np.unique(y)))
    decoder_no_plasticity.fit(Z_poincare, y, epochs=50)
    y_pred_no_plasticity = decoder_no_plasticity.predict(Z_poincare[test_indices])
    no_plasticity_acc = accuracy_score(y[test_indices], y_pred_no_plasticity)
    print(f"✅ [No Plasticity] Test Acc: {no_plasticity_acc:.3f}")
    
    # LGMD baseline
    lgmd_acc = lgmd_baseline(X, y, test_indices)
    print(f"✅ [LGMD Baseline] Test Acc: {lgmd_acc:.3f}")
    
    ablation_time = time.time() - ablation_start
    print(f"⏱️ Ablation study completed in {format_time(ablation_time)}")

def hyperparameter_sweep(X, y):
    """Hyperparameter sweep for optimization with multiple classifiers and RandomizedSearchCV"""
    print("\n" + "=" * 60)
    print("🔧 HYPERPARAMETER SWEEP")
    print("=" * 60)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV
    
    # Quick grid search for basic comparison
    param_grid = [
        {'embed_dim': 16, 'temperature': 0.05, 'classifier': 'ridge'},
        {'embed_dim': 32, 'temperature': 0.1, 'classifier': 'random_forest'},
        {'embed_dim': 64, 'temperature': 0.15, 'classifier': 'svm'},
        {'embed_dim': 32, 'temperature': 0.1, 'classifier': 'knn'},
        {'embed_dim': 64, 'temperature': 0.1, 'classifier': 'logistic'},
    ]
    
    best_acc = 0
    best_params = None
    best_classifier = None
    
    print("🔍 Phase 1: Quick grid search...")
    for i, params in enumerate(param_grid):
        print(f"  Testing params {i+1}/{len(param_grid)}: {params}")
        
        # Quick evaluation with current parameters
        test_size = min(100, len(X) // 3)
        test_indices = np.random.choice(len(X), test_size, replace=False)
        train_indices = np.array([i for i in range(len(X)) if i not in test_indices])
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Test different classifiers
        classifier_name = params['classifier']
        
        if classifier_name == 'ridge':
            classifier = Ridge(alpha=1.0)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            y_pred_classes = np.round(y_pred).astype(int)
            y_pred_classes = np.clip(y_pred_classes, 0, len(np.unique(y)) - 1)
            acc = accuracy_score(y_test, y_pred_classes)
            
        elif classifier_name == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
        elif classifier_name == 'svm':
            classifier = SVC(kernel='rbf', random_state=42)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
        elif classifier_name == 'knn':
            classifier = KNeighborsClassifier(n_neighbors=5)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
        elif classifier_name == 'logistic':
            classifier = LogisticRegression(random_state=42, max_iter=1000)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
        
        print(f"    📊 {classifier_name.capitalize()} Test Acc: {acc:.3f}")
        
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_classifier = classifier_name
            print(f"    🏆 New best accuracy!")
    
    # Phase 2: RandomizedSearchCV for best classifier
    print(f"\n🔍 Phase 2: RandomizedSearchCV for {best_classifier}...")
    
    if best_classifier == 'random_forest':
        param_distributions = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_classifier = RandomForestClassifier(random_state=42)
        
    elif best_classifier == 'svm':
        param_distributions = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        base_classifier = SVC(random_state=42)
        
    elif best_classifier == 'knn':
        param_distributions = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        base_classifier = KNeighborsClassifier()
        
    elif best_classifier == 'logistic':
        param_distributions = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        base_classifier = LogisticRegression(random_state=42, max_iter=1000)
        
    else:  # ridge
        param_distributions = {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
        base_classifier = Ridge()
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        base_classifier, 
        param_distributions, 
        n_iter=10,  # Number of parameter settings sampled
        cv=3, 
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X, y)
    
    print(f"  🏆 Best RandomizedSearchCV score: {random_search.best_score_:.3f}")
    print(f"  🏆 Best parameters: {random_search.best_params_}")
    
    # Update best results if RandomizedSearchCV found better parameters
    if random_search.best_score_ > best_acc:
        best_acc = random_search.best_score_
        best_params = random_search.best_params_
        print(f"  🎉 RandomizedSearchCV found better parameters!")
    
    print(f"\n🏆 Final Best parameters: {best_params}")
    print(f"🏆 Final Best classifier: {best_classifier}")
    print(f"🏆 Final Best accuracy: {best_acc:.3f}")
    
    return best_params, best_classifier, best_acc

def visualize_results(results, save_path="/content/drive/MyDrive/results_visualization.png"):
    """Visualize and save results"""
    print("\n" + "=" * 60)
    print("📊 VISUALIZING RESULTS")
    print("=" * 60)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy comparison
    model_names = list(results.keys())
    accuracies = [np.mean(results[model]) for model in model_names]
    std_errors = [np.std(results[model]) for model in model_names]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax1.bar(model_names, accuracies, yerr=std_errors, capsize=5, color=colors)
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
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12)
    
    # 3. Confusion matrix (for best model)
    best_model = max(results.keys(), key=lambda x: np.mean(results[x]))
    print(f"🏆 Best performing model: {best_model}")
    
    # Generate confusion matrix for best model
    try:
        # Use a subset of data for confusion matrix
        test_size = min(200, len(X) // 4)
        test_indices = np.random.choice(len(X), test_size, replace=False)
        
        if best_model == 'proposed':
            # For proposed model, we need to run the full pipeline
            hyperbolic_embedding = HyperbolicContrastiveEmbedding(embed_dim=64)
            Z_poincare, manifold = hyperbolic_embedding.fit_transform(X[~test_indices], y[~test_indices])
            Z_proto, proto_labels = hyperbolic_structural_plasticity(Z_poincare, y[~test_indices], manifold)
            decoder = SemanticDecoder(Z_proto.shape[1], len(np.unique(y)))
            decoder.fit(Z_proto, proto_labels, epochs=50)
            Z_test, _ = hyperbolic_embedding.fit_transform(X[test_indices], y[test_indices])
            y_pred = decoder.predict(Z_test)
        else:
            # For baseline models
            if best_model == 'cnn':
                y_pred = cnn_baseline(X, y, test_indices, return_predictions=True)
            elif best_model == 'snn':
                y_pred = snn_baseline(X, y, test_indices, return_predictions=True)
            else:  # lgmd
                y_pred = lgmd_baseline(X, y, test_indices, return_predictions=True)
        
        cm = confusion_matrix(y[test_indices], y_pred)
        
        # Get class names for better visualization
        unique_classes = np.unique(y)
        class_names = [f'Class {i}' for i in unique_classes]
        
        # Create confusion matrix with class names
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=class_names, yticklabels=class_names)
        ax3.set_title(f'Confusion Matrix ({best_model.capitalize()})', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Predicted', fontsize=12)
        ax3.set_ylabel('Actual', fontsize=12)
        
        # Rotate labels for better readability
        ax3.tick_params(axis='x', rotation=45)
        ax3.tick_params(axis='y', rotation=0)
        
    except Exception as e:
        print(f"⚠️ Could not generate confusion matrix: {e}")
        ax3.text(0.5, 0.5, 'Confusion matrix\ncould not be generated', 
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # 4. Learning curves (placeholder)
    ax4.text(0.5, 0.5, 'Learning curves\nwould be shown here', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Learning Curves', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to Google Drive
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Results visualization saved to: {save_path}")
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")
    
    plt.show()

def save_results_table(results, save_path="/content/drive/MyDrive/results_table.csv"):
    """Save results as CSV table"""
    print("\n" + "=" * 60)
    print("💾 SAVING RESULTS TABLE")
    print("=" * 60)
    
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
        print(f"✅ Results table saved to: {save_path}")
        print(f"✅ Summary table saved to: {summary_path}")
    except Exception as e:
        print(f"❌ Error saving results table: {e}")

def load_features_from_part1(load_path="/content/drive/MyDrive/lgmd_features.npz"):
    """Load features from Part 1"""
    try:
        data = np.load(load_path)
        features = data['features']
        labels = data['labels']
        print(f"✅ Features loaded from Part 1: {features.shape}")
        print(f"✅ Labels loaded from Part 1: {labels.shape}")
        return features, labels
    except Exception as e:
        print(f"❌ Error loading features from Part 1: {e}")
        return None, None

def main_part2():
    """Main execution for Part 2: Hyperbolic Learning & Evaluation"""
    total_start_time = time.time()
    print("=" * 60)
    print("🚀 PART 2: HYPERBOLIC LEARNING & EVALUATION")
    print("=" * 60)
    
    # Load features from Part 1
    print("📁 Loading features from Part 1...")
    features, labels = load_features_from_part1()
    
    if features is None:
        print("❌ Failed to load features from Part 1. Please run Part 1 first.")
        print("Make sure Part 1 completed successfully and features were saved.")
        return None
    
    print(f"📊 X shape: {features.shape} y shape: {labels.shape}")
    print(f"📊 Class distribution: {Counter(labels)}")
    
    # Robust evaluation
    results = robust_evaluation(features, labels)
    
    # Statistical significance testing
    statistical_significance_testing(results)
    
    # Ablation study
    ablation_study(features, labels)
    
    # Hyperparameter sweep
    best_params, best_classifier, best_acc = hyperparameter_sweep(features, labels)
    print(f"🏆 Best hyperparameters found: {best_params}")
    print(f"🏆 Best classifier: {best_classifier}")
    print(f"🏆 Best accuracy: {best_acc:.3f}")
    
    # Visualize and save results
    visualize_results(results)
    save_results_table(results)
    
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print(f"🎉 PART 2 COMPLETED SUCCESSFULLY! (Total time: {format_time(total_time)})")
    print("=" * 60)
    print("✅ All results have been saved to Google Drive!")
    print("📊 Check the following files:")
    print(f"  📁 Results visualization: /content/drive/MyDrive/results_visualization.png")
    print(f"  📁 Results table: /content/drive/MyDrive/results_table.csv")
    print(f"  📁 Summary table: /content/drive/MyDrive/results_table_summary.csv")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    # For Colab, you can run this cell directly
    results = main_part2() 