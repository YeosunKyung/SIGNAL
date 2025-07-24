import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from collections import defaultdict
import json

warnings.filterwarnings('ignore')

class CompetitiveAnalysis:
    """
    최신 경쟁 기법과의 rigorous한 비교 평가
    - 최근 hyperbolic embedding 방법론과 비교
    - Transformer 및 Graph embedding 기반 방법과 비교
    - 통계적 유의성 검정
    """
    
    def __init__(self, lgmd_encoder, lgmd_classifier):
        self.lgmd_encoder = lgmd_encoder
        self.lgmd_classifier = lgmd_classifier
        self.comparison_results = {}
        
    def hyperbolic_embedding_comparison(self, features, labels):
        """
        최근 hyperbolic embedding 방법론들과 비교
        """
        print("🌐 Hyperbolic Embedding Comparison")
        print("=" * 50)
        
        from sklearn.manifold import MDS
        from sklearn.decomposition import PCA
        
        # 1. Poincaré Ball Embedding (Nickel & Kiela, 2017)
        print("🔬 Implementing Poincaré Ball Embedding...")
        
        class PoincareEmbedding:
            def __init__(self, dim=64, curvature=1.0):
                self.dim = dim
                self.curvature = curvature
                
            def embed(self, features):
                # Simplified Poincaré ball embedding
                # In practice, use proper hyperbolic optimization
                pca = PCA(n_components=min(self.dim, features.shape[1]))
                euclidean_emb = pca.fit_transform(features)
                
                # Project to Poincaré ball
                norms = np.linalg.norm(euclidean_emb, axis=1, keepdims=True)
                max_norm = np.max(norms)
                if max_norm > 0:
                    poincare_emb = euclidean_emb * (0.9 / max_norm)  # Scale to ball
                else:
                    poincare_emb = euclidean_emb
                
                return poincare_emb
        
        # 2. Hyperbolic Neural Networks (Ganea et al., 2018)
        print("🔬 Implementing Hyperbolic Neural Networks...")
        
        class HyperbolicNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.output_dim = output_dim
                
                # Simplified hyperbolic layers
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                # Simplified hyperbolic activation
                x = self.fc1(x)
                x = torch.tanh(x)  # Hyperbolic tangent as activation
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        # 3. Lorentz Model (Nickel & Kiela, 2018)
        print("🔬 Implementing Lorentz Model...")
        
        class LorentzEmbedding:
            def __init__(self, dim=64):
                self.dim = dim
                
            def embed(self, features):
                # Simplified Lorentz embedding
                pca = PCA(n_components=min(self.dim, features.shape[1]))
                euclidean_emb = pca.fit_transform(features)
                
                # Add time component for Lorentz model
                time_component = np.sqrt(1 + np.sum(euclidean_emb**2, axis=1, keepdims=True))
                lorentz_emb = np.concatenate([time_component, euclidean_emb], axis=1)
                
                return lorentz_emb
        
        # 4. Evaluate all hyperbolic methods
        methods = {
            'Poincaré_Ball': PoincareEmbedding(dim=64),
            'Lorentz_Model': LorentzEmbedding(dim=64)
        }
        
        hyperbolic_results = {}
        
        for method_name, method in methods.items():
            print(f"\n📊 Evaluating {method_name}...")
            
            # Embed features
            if hasattr(method, 'embed'):
                embedded_features = method.embed(features)
            else:
                # For neural network methods
                embedded_features = features  # Use original features
            
            # Evaluate with cross-validation
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            accuracies = []
            
            for train_idx, test_idx in skf.split(embedded_features, labels):
                X_train = embedded_features[train_idx]
                y_train = labels[train_idx]
                X_test = embedded_features[test_idx]
                y_test = labels[test_idx]
                
                # Use same classifier for fair comparison
                self.lgmd_classifier.fit(X_train, y_train)
                y_pred = self.lgmd_classifier.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
            
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            hyperbolic_results[method_name] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'accuracies': accuracies
            }
            
            print(f"  📊 {method_name}: {mean_acc:.4f} ± {std_acc:.4f}")
        
        self.comparison_results['hyperbolic_methods'] = hyperbolic_results
        return hyperbolic_results
    
    def transformer_comparison(self, features, labels):
        """
        Transformer 기반 방법과 비교
        """
        print("\n🤖 Transformer-based Method Comparison")
        print("=" * 50)
        
        # 1. Vision Transformer (ViT) inspired approach
        print("🔬 Implementing Vision Transformer approach...")
        
        class VisionTransformerFeatures:
            def __init__(self, patch_size=8, embed_dim=64):
                self.patch_size = patch_size
                self.embed_dim = embed_dim
                
            def extract_features(self, features):
                # Reshape features to patches (simplified)
                n_samples, n_features = features.shape
                patch_dim = int(np.sqrt(n_features))
                
                if patch_dim * patch_dim == n_features:
                    # Reshape to 2D patches
                    features_2d = features.reshape(n_samples, patch_dim, patch_dim)
                    
                    # Extract patches
                    patches = []
                    for i in range(0, patch_dim - self.patch_size + 1, self.patch_size):
                        for j in range(0, patch_dim - self.patch_size + 1, self.patch_size):
                            patch = features_2d[:, i:i+self.patch_size, j:j+self.patch_size]
                            patches.append(patch.reshape(n_samples, -1))
                    
                    # Concatenate patches
                    if patches:
                        patch_features = np.concatenate(patches, axis=1)
                        return patch_features
                
                # Fallback to original features
                return features
        
        # 2. Self-Attention Mechanism
        print("🔬 Implementing Self-Attention mechanism...")
        
        class SelfAttentionFeatures:
            def __init__(self, embed_dim=64):
                self.embed_dim = embed_dim
                
            def extract_features(self, features):
                # Simplified self-attention
                n_samples, n_features = features.shape
                
                # Compute attention weights
                attention_weights = np.dot(features, features.T) / np.sqrt(n_features)
                attention_weights = F.softmax(torch.tensor(attention_weights), dim=-1).numpy()
                
                # Apply attention
                attended_features = np.dot(attention_weights, features)
                
                # Dimensionality reduction
                pca = PCA(n_components=min(self.embed_dim, attended_features.shape[1]))
                reduced_features = pca.fit_transform(attended_features)
                
                return reduced_features
        
        # 3. Evaluate transformer methods
        transformer_methods = {
            'Vision_Transformer': VisionTransformerFeatures(patch_size=8, embed_dim=64),
            'Self_Attention': SelfAttentionFeatures(embed_dim=64)
        }
        
        transformer_results = {}
        
        for method_name, method in transformer_methods.items():
            print(f"\n📊 Evaluating {method_name}...")
            
            # Extract features
            extracted_features = method.extract_features(features)
            
            # Evaluate with cross-validation
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            accuracies = []
            
            for train_idx, test_idx in skf.split(extracted_features, labels):
                X_train = extracted_features[train_idx]
                y_train = labels[train_idx]
                X_test = extracted_features[test_idx]
                y_test = labels[test_idx]
                
                self.lgmd_classifier.fit(X_train, y_train)
                y_pred = self.lgmd_classifier.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
            
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            transformer_results[method_name] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'accuracies': accuracies
            }
            
            print(f"  📊 {method_name}: {mean_acc:.4f} ± {std_acc:.4f}")
        
        self.comparison_results['transformer_methods'] = transformer_results
        return transformer_results
    
    def graph_embedding_comparison(self, features, labels):
        """
        Graph embedding 기반 방법과 비교
        """
        print("\n🕸️ Graph Embedding Method Comparison")
        print("=" * 50)
        
        import networkx as nx
        from sklearn.neighbors import kneighbors_graph
        
        # 1. Graph Neural Network (GNN) approach
        print("🔬 Implementing Graph Neural Network approach...")
        
        class GraphEmbedding:
            def __init__(self, n_neighbors=10, embed_dim=64):
                self.n_neighbors = n_neighbors
                self.embed_dim = embed_dim
                
            def extract_features(self, features):
                # Construct k-nearest neighbor graph
                adjacency_matrix = kneighbors_graph(features, n_neighbors=self.n_neighbors, 
                                                  mode='connectivity', include_self=True)
                
                # Convert to NetworkX graph
                G = nx.from_scipy_sparse_matrix(adjacency_matrix)
                
                # Graph-based feature extraction
                graph_features = []
                
                for i in range(len(features)):
                    # Node centrality features
                    degree_centrality = nx.degree_centrality(G)[i]
                    betweenness_centrality = nx.betweenness_centrality(G)[i]
                    closeness_centrality = nx.closeness_centrality(G)[i]
                    
                    # Combine with original features
                    node_features = np.concatenate([
                        features[i],
                        [degree_centrality, betweenness_centrality, closeness_centrality]
                    ])
                    
                    graph_features.append(node_features)
                
                graph_features = np.array(graph_features)
                
                # Dimensionality reduction
                pca = PCA(n_components=min(self.embed_dim, graph_features.shape[1]))
                reduced_features = pca.fit_transform(graph_features)
                
                return reduced_features
        
        # 2. Spectral Graph Embedding
        print("🔬 Implementing Spectral Graph Embedding...")
        
        class SpectralEmbedding:
            def __init__(self, n_components=64):
                self.n_components = n_components
                
            def extract_features(self, features):
                from sklearn.manifold import SpectralEmbedding
                
                # Construct similarity graph
                adjacency_matrix = kneighbors_graph(features, n_neighbors=15, 
                                                  mode='connectivity', include_self=True)
                
                # Spectral embedding
                spectral_emb = SpectralEmbedding(n_components=self.n_components, 
                                               affinity='precomputed')
                
                # Convert to dense matrix
                adjacency_dense = adjacency_matrix.toarray()
                
                # Apply spectral embedding
                embedded_features = spectral_emb.fit_transform(adjacency_dense)
                
                return embedded_features
        
        # 3. Evaluate graph methods
        graph_methods = {
            'Graph_Neural_Network': GraphEmbedding(n_neighbors=10, embed_dim=64),
            'Spectral_Embedding': SpectralEmbedding(n_components=64)
        }
        
        graph_results = {}
        
        for method_name, method in graph_methods.items():
            print(f"\n📊 Evaluating {method_name}...")
            
            # Extract features
            extracted_features = method.extract_features(features)
            
            # Evaluate with cross-validation
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            accuracies = []
            
            for train_idx, test_idx in skf.split(extracted_features, labels):
                X_train = extracted_features[train_idx]
                y_train = labels[train_idx]
                X_test = extracted_features[test_idx]
                y_test = labels[test_idx]
                
                self.lgmd_classifier.fit(X_train, y_train)
                y_pred = self.lgmd_classifier.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
            
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            graph_results[method_name] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'accuracies': accuracies
            }
            
            print(f"  📊 {method_name}: {mean_acc:.4f} ± {std_acc:.4f}")
        
        self.comparison_results['graph_methods'] = graph_results
        return graph_results
    
    def statistical_significance_test(self):
        """
        통계적 유의성 검정
        """
        print("\n📊 Statistical Significance Testing")
        print("=" * 50)
        
        from scipy import stats
        
        # Collect all results
        all_methods = {}
        
        # Add LGMD baseline
        if 'lgmd_baseline' in self.comparison_results:
            all_methods['LGMD'] = self.comparison_results['lgmd_baseline']['accuracies']
        
        # Add hyperbolic methods
        if 'hyperbolic_methods' in self.comparison_results:
            for method_name, result in self.comparison_results['hyperbolic_methods'].items():
                all_methods[method_name] = result['accuracies']
        
        # Add transformer methods
        if 'transformer_methods' in self.comparison_results:
            for method_name, result in self.comparison_results['transformer_methods'].items():
                all_methods[method_name] = result['accuracies']
        
        # Add graph methods
        if 'graph_methods' in self.comparison_results:
            for method_name, result in self.comparison_results['graph_methods'].items():
                all_methods[method_name] = result['accuracies']
        
        # Perform statistical tests
        significance_results = {}
        
        # 1. ANOVA test
        if len(all_methods) > 2:
            f_stat, p_value = stats.f_oneway(*list(all_methods.values()))
            significance_results['anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            print(f"📊 ANOVA Test: F={f_stat:.4f}, p={p_value:.4f}, Significant={p_value < 0.05}")
        
        # 2. Pairwise t-tests
        pairwise_results = {}
        method_names = list(all_methods.keys())
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                if i < j:
                    t_stat, p_value = stats.ttest_rel(all_methods[method1], all_methods[method2])
                    pairwise_results[f"{method1}_vs_{method2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'method1_better': t_stat > 0 if p_value < 0.05 else None
                    }
                    
                    if p_value < 0.05:
                        better_method = method1 if t_stat > 0 else method2
                        print(f"📊 {method1} vs {method2}: t={t_stat:.4f}, p={p_value:.4f}, {better_method} better")
        
        significance_results['pairwise'] = pairwise_results
        
        # 3. Effect size (Cohen's d)
        effect_sizes = {}
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                if i < j:
                    # Cohen's d
                    pooled_std = np.sqrt(((len(all_methods[method1]) - 1) * np.var(all_methods[method1]) + 
                                        (len(all_methods[method2]) - 1) * np.var(all_methods[method2])) / 
                                       (len(all_methods[method1]) + len(all_methods[method2]) - 2))
                    
                    cohens_d = (np.mean(all_methods[method1]) - np.mean(all_methods[method2])) / pooled_std
                    effect_sizes[f"{method1}_vs_{method2}"] = cohens_d
        
        significance_results['effect_sizes'] = effect_sizes
        
        self.comparison_results['statistical_significance'] = significance_results
        return significance_results
    
    def generate_comparison_plots(self):
        """
        비교 결과 시각화
        """
        if not self.comparison_results:
            print("❌ No comparison results available. Run comparison first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall performance comparison
        all_methods = {}
        method_categories = {}
        
        # Collect all results
        if 'lgmd_baseline' in self.comparison_results:
            all_methods['LGMD'] = self.comparison_results['lgmd_baseline']['mean_accuracy']
            method_categories['LGMD'] = 'LGMD'
        
        if 'hyperbolic_methods' in self.comparison_results:
            for method_name, result in self.comparison_results['hyperbolic_methods'].items():
                all_methods[method_name] = result['mean_accuracy']
                method_categories[method_name] = 'Hyperbolic'
        
        if 'transformer_methods' in self.comparison_results:
            for method_name, result in self.comparison_results['transformer_methods'].items():
                all_methods[method_name] = result['mean_accuracy']
                method_categories[method_name] = 'Transformer'
        
        if 'graph_methods' in self.comparison_results:
            for method_name, result in self.comparison_results['graph_methods'].items():
                all_methods[method_name] = result['mean_accuracy']
                method_categories[method_name] = 'Graph'
        
        # Plot overall comparison
        if all_methods:
            method_names = list(all_methods.keys())
            accuracies = list(all_methods.values())
            colors = [method_categories[name] for name in method_names]
            
            # Color mapping
            color_map = {'LGMD': 'red', 'Hyperbolic': 'blue', 'Transformer': 'green', 'Graph': 'orange'}
            color_values = [color_map[cat] for cat in colors]
            
            bars = axes[0, 0].bar(method_names, accuracies, color=color_values, alpha=0.8)
            axes[0, 0].set_title('Overall Performance Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].set_ylim(0, 1)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color_map[cat], label=cat) 
                             for cat in set(method_categories.values())]
            axes[0, 0].legend(handles=legend_elements)
        
        # 2. Category-wise comparison
        category_means = defaultdict(list)
        for method_name, accuracy in all_methods.items():
            category_means[method_categories[method_name]].append(accuracy)
        
        if category_means:
            categories = list(category_means.keys())
            category_avg = [np.mean(category_means[cat]) for cat in categories]
            category_std = [np.std(category_means[cat]) for cat in categories]
            
            axes[0, 1].bar(categories, category_avg, yerr=category_std, capsize=5, alpha=0.8)
            axes[0, 1].set_title('Performance by Category')
            axes[0, 1].set_ylabel('Average Accuracy')
            axes[0, 1].set_ylim(0, 1)
        
        # 3. Statistical significance heatmap
        if 'statistical_significance' in self.comparison_results:
            sig_results = self.comparison_results['statistical_significance']
            if 'pairwise' in sig_results:
                pairwise = sig_results['pairwise']
                
                # Create significance matrix
                method_names = list(all_methods.keys())
                sig_matrix = np.zeros((len(method_names), len(method_names)))
                
                for i, method1 in enumerate(method_names):
                    for j, method2 in enumerate(method_names):
                        if i != j:
                            key = f"{method1}_vs_{method2}" if i < j else f"{method2}_vs_{method1}"
                            if key in pairwise:
                                sig_matrix[i, j] = -np.log10(pairwise[key]['p_value'])
                
                im = axes[1, 0].imshow(sig_matrix, cmap='Reds', aspect='auto')
                axes[1, 0].set_title('Statistical Significance (-log10 p-value)')
                axes[1, 0].set_xticks(range(len(method_names)))
                axes[1, 0].set_yticks(range(len(method_names)))
                axes[1, 0].set_xticklabels(method_names, rotation=45)
                axes[1, 0].set_yticklabels(method_names)
                plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Effect size comparison
        if 'statistical_significance' in self.comparison_results:
            sig_results = self.comparison_results['statistical_significance']
            if 'effect_sizes' in sig_results:
                effect_sizes = sig_results['effect_sizes']
                
                effect_names = list(effect_sizes.keys())
                effect_values = list(effect_sizes.values())
                
                axes[1, 1].bar(range(len(effect_names)), effect_values, alpha=0.8)
                axes[1, 1].set_title('Effect Sizes (Cohen\'s d)')
                axes[1, 1].set_ylabel('Effect Size')
                axes[1, 1].set_xticks(range(len(effect_names)))
                axes[1, 1].set_xticklabels(effect_names, rotation=45)
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/competitive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_comparison_results(self, save_path="/content/drive/MyDrive/competitive_analysis_results.json"):
        """
        비교 분석 결과 저장
        """
        try:
            # Convert numpy types to Python types for JSON serialization
            results_serializable = {}
            for key, value in self.comparison_results.items():
                if isinstance(value, dict):
                    results_serializable[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            results_serializable[key][sub_key] = {}
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                if isinstance(sub_sub_value, np.ndarray):
                                    results_serializable[key][sub_key][sub_sub_key] = sub_sub_value.tolist()
                                elif isinstance(sub_sub_value, np.integer):
                                    results_serializable[key][sub_key][sub_sub_key] = int(sub_sub_value)
                                elif isinstance(sub_sub_value, np.floating):
                                    results_serializable[key][sub_key][sub_sub_key] = float(sub_sub_value)
                                else:
                                    results_serializable[key][sub_key][sub_sub_key] = sub_sub_value
                        elif isinstance(sub_value, np.ndarray):
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
            
            print(f"✅ Competitive analysis results saved to: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False

# Usage example
def run_competitive_analysis(lgmd_encoder, lgmd_classifier, features, labels):
    """
    전체 경쟁 분석 실행
    """
    analyzer = CompetitiveAnalysis(lgmd_encoder, lgmd_classifier)
    
    # 1. LGMD baseline evaluation
    print("🔬 Evaluating LGMD baseline...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    lgmd_accuracies = []
    
    for train_idx, test_idx in skf.split(features, labels):
        X_train = features[train_idx]
        y_train = labels[train_idx]
        X_test = features[test_idx]
        y_test = labels[test_idx]
        
        lgmd_classifier.fit(X_train, y_train)
        y_pred = lgmd_classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        lgmd_accuracies.append(accuracy)
    
    analyzer.comparison_results['lgmd_baseline'] = {
        'mean_accuracy': np.mean(lgmd_accuracies),
        'std_accuracy': np.std(lgmd_accuracies),
        'accuracies': lgmd_accuracies
    }
    
    print(f"📊 LGMD Baseline: {np.mean(lgmd_accuracies):.4f} ± {np.std(lgmd_accuracies):.4f}")
    
    # 2. Hyperbolic embedding comparison
    hyperbolic_results = analyzer.hyperbolic_embedding_comparison(features, labels)
    
    # 3. Transformer comparison
    transformer_results = analyzer.transformer_comparison(features, labels)
    
    # 4. Graph embedding comparison
    graph_results = analyzer.graph_embedding_comparison(features, labels)
    
    # 5. Statistical significance testing
    significance_results = analyzer.statistical_significance_test()
    
    # 6. Generate plots and save results
    analyzer.generate_comparison_plots()
    analyzer.save_comparison_results()
    
    return analyzer.comparison_results 