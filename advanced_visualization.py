import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

class AdvancedVisualization:
    """
    고급 시각화 및 분석 모듈
    - Hyperbolic embedding의 class별 분포 시각화
    - 구조적 가소성 프로토타입 선택 과정 시각화
    - Loss curve 및 Learning curve 시각화
    """
    
    def __init__(self, lgmd_encoder, lgmd_classifier):
        self.lgmd_encoder = lgmd_encoder
        self.lgmd_classifier = lgmd_classifier
        self.visualization_results = {}
        
    def hyperbolic_embedding_visualization(self, features, labels, class_names=None):
        """
        Hyperbolic embedding의 class별 분포 시각화
        """
        print("🌐 Hyperbolic Embedding Visualization")
        print("=" * 50)
        
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(len(np.unique(labels)))]
        
        # 1. Poincaré Ball Embedding
        print("🔬 Creating Poincaré Ball embedding...")
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=min(50, features.shape[1]))
        reduced_features = pca.fit_transform(features)
        
        # Project to Poincaré ball
        norms = np.linalg.norm(reduced_features, axis=1, keepdims=True)
        max_norm = np.max(norms)
        if max_norm > 0:
            poincare_emb = reduced_features * (0.9 / max_norm)  # Scale to ball
        else:
            poincare_emb = reduced_features
        
        # 2. Create interactive Poincaré ball visualization
        fig = go.Figure()
        
        # Add unit circle boundary
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=circle_x, y=circle_y,
            mode='lines',
            line=dict(color='black', width=2),
            name='Poincaré Ball Boundary',
            showlegend=False
        ))
        
        # Add data points colored by class
        for class_idx in np.unique(labels):
            class_mask = labels == class_idx
            class_points = poincare_emb[class_mask]
            
            fig.add_trace(go.Scatter(
                x=class_points[:, 0], y=class_points[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=class_idx,
                    colorscale='Viridis',
                    opacity=0.7
                ),
                name=class_names[class_idx],
                text=[f'{class_names[class_idx]}<br>Point {i}' for i in range(len(class_points))],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Hyperbolic Embedding in Poincaré Ball',
            xaxis_title='X',
            yaxis_title='Y',
            width=800, height=600,
            showlegend=True
        )
        
        # Save as HTML for interactive viewing
        fig.write_html('/content/drive/MyDrive/hyperbolic_embedding.html')
        
        # 3. Create static matplotlib version
        plt.figure(figsize=(12, 8))
        
        # Plot unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        plt.gca().add_patch(circle)
        
        # Plot data points
        for class_idx in np.unique(labels):
            class_mask = labels == class_idx
            class_points = poincare_emb[class_mask]
            
            plt.scatter(class_points[:, 0], class_points[:, 1], 
                       label=class_names[class_idx], alpha=0.7, s=50)
        
        plt.title('Hyperbolic Embedding in Poincaré Ball', fontsize=16)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        
        plt.savefig('/content/drive/MyDrive/hyperbolic_embedding_static.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Distance analysis in hyperbolic space
        print("📊 Analyzing hyperbolic distances...")
        
        # Calculate hyperbolic distances
        hyperbolic_distances = self.calculate_hyperbolic_distances(poincare_emb)
        
        # Visualize distance distributions
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Intra-class vs inter-class distances
        intra_distances = []
        inter_distances = []
        
        for class_idx in np.unique(labels):
            class_mask = labels == class_idx
            other_mask = labels != class_idx
            
            class_points = poincare_emb[class_mask]
            other_points = poincare_emb[other_mask]
            
            # Intra-class distances
            if len(class_points) > 1:
                intra_dist = hyperbolic_distances[np.ix_(class_mask, class_mask)]
                intra_distances.extend(intra_dist[np.triu_indices_from(intra_dist, k=1)])
            
            # Inter-class distances
            if len(other_points) > 0:
                inter_dist = hyperbolic_distances[np.ix_(class_mask, other_mask)]
                inter_distances.extend(inter_dist.flatten())
        
        axes[0].hist(intra_distances, bins=30, alpha=0.7, label='Intra-class', density=True)
        axes[0].hist(inter_distances, bins=30, alpha=0.7, label='Inter-class', density=True)
        axes[0].set_title('Distance Distribution in Hyperbolic Space')
        axes[0].set_xlabel('Hyperbolic Distance')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        
        # Class separation analysis
        separation_scores = []
        for class_idx in np.unique(labels):
            class_mask = labels == class_idx
            other_mask = labels != class_idx
            
            if np.sum(class_mask) > 0 and np.sum(other_mask) > 0:
                class_center = np.mean(poincare_emb[class_mask], axis=0)
                other_center = np.mean(poincare_emb[other_mask], axis=0)
                
                separation = np.linalg.norm(class_center - other_center)
                separation_scores.append(separation)
        
        axes[1].bar(range(len(separation_scores)), separation_scores)
        axes[1].set_title('Class Separation in Hyperbolic Space')
        axes[1].set_xlabel('Class Index')
        axes[1].set_ylabel('Separation Score')
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/hyperbolic_distance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.visualization_results['hyperbolic_embedding'] = {
            'poincare_embedding': poincare_emb,
            'hyperbolic_distances': hyperbolic_distances,
            'separation_scores': separation_scores
        }
        
        return poincare_emb, hyperbolic_distances
    
    def calculate_hyperbolic_distances(self, poincare_emb):
        """
        Poincaré ball에서의 쌍곡선 거리 계산
        """
        n_points = len(poincare_emb)
        distances = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                # Poincaré distance formula
                diff = poincare_emb[i] - poincare_emb[j]
                norm_diff = np.linalg.norm(diff)
                norm_i = np.linalg.norm(poincare_emb[i])
                norm_j = np.linalg.norm(poincare_emb[j])
                
                if norm_diff == 0:
                    distances[i, j] = distances[j, i] = 0
                else:
                    # Poincaré distance
                    numerator = 2 * norm_diff**2
                    denominator = (1 - norm_i**2) * (1 - norm_j**2)
                    distances[i, j] = distances[j, i] = np.arccosh(1 + numerator / denominator)
        
        return distances
    
    def structural_plasticity_visualization(self, features, labels, plasticity_params=None):
        """
        구조적 가소성 프로토타입 선택 과정 시각화
        """
        print("\n🧠 Structural Plasticity Visualization")
        print("=" * 50)
        
        if plasticity_params is None:
            plasticity_params = {
                'feature_weights': np.ones(features.shape[1]),
                'prototype_selection': 'adaptive',
                'plasticity_threshold': 0.1
            }
        
        # 1. Feature importance evolution
        print("📊 Visualizing feature importance evolution...")
        
        # Simulate feature importance evolution over time
        n_iterations = 50
        feature_importance_history = []
        
        # Initialize with random importance
        current_importance = np.random.rand(features.shape[1])
        current_importance = current_importance / np.sum(current_importance)
        
        for iteration in range(n_iterations):
            # Simulate plasticity update
            # In practice, this would be based on actual learning dynamics
            
            # Add noise and update based on class separability
            noise = np.random.normal(0, 0.01, current_importance.shape)
            current_importance += noise
            
            # Update based on feature-class correlation
            for i in range(features.shape[1]):
                feature_values = features[:, i]
                # Calculate correlation with labels
                correlation = np.corrcoef(feature_values, labels)[0, 1]
                if not np.isnan(correlation):
                    current_importance[i] += 0.001 * abs(correlation)
            
            # Normalize
            current_importance = np.maximum(current_importance, 0)
            current_importance = current_importance / np.sum(current_importance)
            
            feature_importance_history.append(current_importance.copy())
        
        feature_importance_history = np.array(feature_importance_history)
        
        # 2. Create evolution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature importance evolution
        for i in range(min(10, features.shape[1])):  # Show top 10 features
            axes[0, 0].plot(feature_importance_history[:, i], 
                           label=f'Feature {i}', alpha=0.7)
        
        axes[0, 0].set_title('Feature Importance Evolution')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Importance Weight')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Final feature importance
        final_importance = feature_importance_history[-1]
        top_features = np.argsort(final_importance)[-10:]  # Top 10 features
        
        axes[0, 1].bar(range(len(top_features)), final_importance[top_features])
        axes[0, 1].set_title('Final Feature Importance (Top 10)')
        axes[0, 1].set_xlabel('Feature Index')
        axes[0, 1].set_ylabel('Importance Weight')
        axes[0, 1].set_xticks(range(len(top_features)))
        axes[0, 1].set_xticklabels([f'F{i}' for i in top_features])
        
        # 3. Prototype selection visualization
        print("🔬 Visualizing prototype selection process...")
        
        # Simulate prototype selection
        n_prototypes = 5
        prototype_indices = self.select_prototypes(features, labels, n_prototypes, final_importance)
        
        # Visualize prototypes in feature space
        if features.shape[1] >= 2:
            # Use first two dimensions for visualization
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            prototypes_2d = features_2d[prototype_indices]
            
            axes[1, 0].scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=labels, alpha=0.6, s=20)
            axes[1, 0].scatter(prototypes_2d[:, 0], prototypes_2d[:, 1], 
                              c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                              label='Prototypes')
            axes[1, 0].set_title('Prototype Selection in Feature Space')
            axes[1, 0].set_xlabel('PC1')
            axes[1, 0].set_ylabel('PC2')
            axes[1, 0].legend()
        
        # 4. Plasticity convergence analysis
        convergence_metric = np.std(feature_importance_history, axis=1)
        axes[1, 1].plot(convergence_metric)
        axes[1, 1].set_title('Plasticity Convergence')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Standard Deviation of Weights')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/structural_plasticity_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Interactive prototype selection
        fig = go.Figure()
        
        if features.shape[1] >= 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            prototypes_2d = features_2d[prototype_indices]
            
            # Add all data points
            for class_idx in np.unique(labels):
                class_mask = labels == class_idx
                class_points = features_2d[class_mask]
                
                fig.add_trace(go.Scatter(
                    x=class_points[:, 0], y=class_points[:, 1],
                    mode='markers',
                    marker=dict(size=8, opacity=0.6),
                    name=f'Class {class_idx}',
                    showlegend=True
                ))
            
            # Add prototypes
            fig.add_trace(go.Scatter(
                x=prototypes_2d[:, 0], y=prototypes_2d[:, 1],
                mode='markers',
                marker=dict(size=20, symbol='star', color='red'),
                name='Prototypes',
                showlegend=True
            ))
        
        fig.update_layout(
            title='Interactive Prototype Selection',
            xaxis_title='PC1',
            yaxis_title='PC2',
            width=800, height=600
        )
        
        fig.write_html('/content/drive/MyDrive/prototype_selection.html')
        
        self.visualization_results['structural_plasticity'] = {
            'feature_importance_history': feature_importance_history,
            'final_importance': final_importance,
            'prototype_indices': prototype_indices,
            'convergence_metric': convergence_metric
        }
        
        return feature_importance_history, prototype_indices
    
    def select_prototypes(self, features, labels, n_prototypes, feature_importance):
        """
        프로토타입 선택
        """
        # Weight features by importance
        weighted_features = features * feature_importance.reshape(1, -1)
        
        # Use K-means to select prototypes
        kmeans = KMeans(n_clusters=n_prototypes, random_state=42)
        cluster_labels = kmeans.fit_predict(weighted_features)
        
        # Select points closest to cluster centers
        prototype_indices = []
        for i in range(n_prototypes):
            cluster_points = np.where(cluster_labels == i)[0]
            if len(cluster_points) > 0:
                cluster_center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(weighted_features[cluster_points] - cluster_center, axis=1)
                closest_idx = cluster_points[np.argmin(distances)]
                prototype_indices.append(closest_idx)
        
        return np.array(prototype_indices)
    
    def loss_and_learning_curves(self, features, labels):
        """
        Loss curve 및 Learning curve 시각화
        """
        print("\n📈 Loss and Learning Curves Visualization")
        print("=" * 50)
        
        # 1. Simulate training process
        print("🔄 Simulating training process...")
        
        n_epochs = 100
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # Split data for training/validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Initialize model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Simulate incremental training
        for epoch in tqdm(range(n_epochs), desc="Training simulation"):
            # Use subset of data for incremental learning simulation
            subset_size = min(len(X_train), int(len(X_train) * (epoch + 1) / n_epochs))
            subset_indices = np.random.choice(len(X_train), subset_size, replace=False)
            
            X_subset = X_train[subset_indices]
            y_subset = y_train[subset_indices]
            
            # Train model
            model.fit(X_subset, y_subset)
            
            # Calculate losses and accuracies
            train_pred = model.predict(X_subset)
            val_pred = model.predict(X_val)
            
            train_acc = np.mean(train_pred == y_subset)
            val_acc = np.mean(val_pred == y_val)
            
            # Simulate loss (1 - accuracy for simplicity)
            train_loss = 1 - train_acc
            val_loss = 1 - val_acc
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
        
        # 2. Create comprehensive learning curves
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        axes[0, 0].plot(train_losses, label='Training Loss', color='blue', alpha=0.8)
        axes[0, 0].plot(val_losses, label='Validation Loss', color='red', alpha=0.8)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(train_accuracies, label='Training Accuracy', color='blue', alpha=0.8)
        axes[0, 1].plot(val_accuracies, label='Validation Accuracy', color='red', alpha=0.8)
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate analysis (simulated)
        learning_rates = np.diff(train_losses)
        axes[0, 2].plot(learning_rates, color='green', alpha=0.8)
        axes[0, 2].set_title('Learning Rate (Loss Gradient)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 3. Advanced analysis
        # Overfitting analysis
        overfitting_gap = np.array(train_accuracies) - np.array(val_accuracies)
        axes[1, 0].plot(overfitting_gap, color='purple', alpha=0.8)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Overfitting Analysis (Train-Val Gap)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy Gap')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Convergence analysis
        convergence_window = 10
        convergence_metric = []
        for i in range(convergence_window, len(val_accuracies)):
            recent_accuracies = val_accuracies[i-convergence_window:i]
            convergence_metric.append(np.std(recent_accuracies))
        
        axes[1, 1].plot(range(convergence_window, len(val_accuracies)), convergence_metric, color='orange', alpha=0.8)
        axes[1, 1].set_title('Convergence Analysis (Validation Stability)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Standard Deviation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance summary
        final_train_acc = train_accuracies[-1]
        final_val_acc = val_accuracies[-1]
        best_val_acc = max(val_accuracies)
        best_epoch = np.argmax(val_accuracies)
        
        summary_text = f"""
        Final Training Accuracy: {final_train_acc:.4f}
        Final Validation Accuracy: {final_val_acc:.4f}
        Best Validation Accuracy: {best_val_acc:.4f}
        Best Epoch: {best_epoch}
        Overfitting: {'Yes' if final_train_acc - final_val_acc > 0.05 else 'No'}
        """
        
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_title('Performance Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Interactive learning curves
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss Curves', 'Accuracy Curves', 'Learning Rate', 'Overfitting Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Loss curves
        fig.add_trace(
            go.Scatter(x=list(range(n_epochs)), y=train_losses, name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(range(n_epochs)), y=val_losses, name='Validation Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Accuracy curves
        fig.add_trace(
            go.Scatter(x=list(range(n_epochs)), y=train_accuracies, name='Training Accuracy', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=list(range(n_epochs)), y=val_accuracies, name='Validation Accuracy', line=dict(color='red')),
            row=1, col=2
        )
        
        # Learning rate
        fig.add_trace(
            go.Scatter(x=list(range(1, n_epochs)), y=learning_rates, name='Learning Rate', line=dict(color='green')),
            row=2, col=1
        )
        
        # Overfitting analysis
        fig.add_trace(
            go.Scatter(x=list(range(n_epochs)), y=overfitting_gap, name='Overfitting Gap', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Interactive Learning Curves")
        fig.write_html('/content/drive/MyDrive/interactive_learning_curves.html')
        
        self.visualization_results['learning_curves'] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'learning_rates': learning_rates,
            'overfitting_gap': overfitting_gap,
            'convergence_metric': convergence_metric,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc
        }
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def comprehensive_analysis_dashboard(self, features, labels, class_names=None):
        """
        종합 분석 대시보드 생성
        """
        print("\n📊 Creating Comprehensive Analysis Dashboard")
        print("=" * 50)
        
        # Run all visualizations
        poincare_emb, hyperbolic_distances = self.hyperbolic_embedding_visualization(features, labels, class_names)
        feature_history, prototype_indices = self.structural_plasticity_visualization(features, labels)
        train_losses, val_losses, train_accs, val_accs = self.loss_and_learning_curves(features, labels)
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Hyperbolic embedding (top left)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        ax1.add_patch(circle)
        
        for class_idx in np.unique(labels):
            class_mask = labels == class_idx
            class_points = poincare_emb[class_mask]
            ax1.scatter(class_points[:, 0], class_points[:, 1], alpha=0.7, s=30)
        
        ax1.set_title('Hyperbolic Embedding', fontsize=14)
        ax1.set_xlim(-1.1, 1.1)
        ax1.set_ylim(-1.1, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # 2. Learning curves (top right)
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        ax2.plot(train_accs, label='Train', color='blue', alpha=0.8)
        ax2.plot(val_accs, label='Validation', color='red', alpha=0.8)
        ax2.set_title('Learning Curves', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature importance evolution (bottom left)
        ax3 = fig.add_subplot(gs[2:4, 0:2])
        feature_history = self.visualization_results['structural_plasticity']['feature_importance_history']
        for i in range(min(5, features.shape[1])):
            ax3.plot(feature_history[:, i], alpha=0.7, label=f'Feature {i}')
        ax3.set_title('Feature Importance Evolution', fontsize=14)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Importance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance metrics (bottom right)
        ax4 = fig.add_subplot(gs[2:4, 2:4])
        
        # Create performance summary
        metrics = {
            'Best Val Acc': max(val_accs),
            'Final Train Acc': train_accs[-1],
            'Final Val Acc': val_accs[-1],
            'Overfitting': train_accs[-1] - val_accs[-1],
            'Convergence': len(val_accs) - np.argmax(val_accs)
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax4.bar(metric_names, metric_values, alpha=0.8)
        ax4.set_title('Performance Metrics', fontsize=14)
        ax4.set_ylabel('Value')
        ax4.tick_params(axis='x', rotation=45)
        
        # Color bars based on values
        for bar, value in zip(bars, metric_values):
            if 'Acc' in bar:
                color = 'green' if value > 0.8 else 'orange' if value > 0.6 else 'red'
            elif 'Overfitting' in bar:
                color = 'red' if value > 0.05 else 'green'
            else:
                color = 'blue'
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save all results
        self.save_visualization_results()
        
        return self.visualization_results
    
    def save_visualization_results(self, save_path="/content/drive/MyDrive/visualization_results.json"):
        """
        시각화 결과 저장
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            results_serializable = {}
            for key, value in self.visualization_results.items():
                if isinstance(value, dict):
                    results_serializable[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            results_serializable[key][sub_key] = sub_value.tolist()
                        elif isinstance(sub_value, np.integer):
                            results_serializable[key][sub_key] = int(sub_value)
                        elif isinstance(sub_value, np.floating):
                            results_serializable[key][sub_key] = float(sub_value)
                        else:
                            results_serializable[key][sub_key] = sub_value
                else:
                    results_serializable[key] = value
            
            with open(save_path, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            print(f"✅ Visualization results saved to: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False

# Usage example
def run_advanced_visualization(lgmd_encoder, lgmd_classifier, features, labels, class_names=None):
    """
    전체 고급 시각화 실행
    """
    visualizer = AdvancedVisualization(lgmd_encoder, lgmd_classifier)
    
    # Run comprehensive analysis
    results = visualizer.comprehensive_analysis_dashboard(features, labels, class_names)
    
    return results 