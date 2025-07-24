import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
import warnings
import os
from collections import Counter
import cv2
from sklearn.preprocessing import StandardScaler
import glob
from scipy.ndimage import gaussian_filter, convolve
import time
from IPython.display import clear_output

# Quick evaluation configuration
QUICK_CONFIG = {
    'use_simple_classifiers': True,
    'max_iterations': 100,
    'timeout_minutes': 30
}

warnings.filterwarnings('ignore')

def quick_simple_evaluation(X, y):
    """Quick evaluation with simple classifiers to avoid hanging"""
    print("=" * 60)
    print("🚀 QUICK SIMPLE EVALUATION (Fast Version)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Simple cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        fold_start = time.time()
        print(f"\n🔄 Fold {fold}/3")
        
        # Prepare data
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        # Simple normalization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Use only fast, reliable classifiers
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        
        classifiers = [
            ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('KNN', KNeighborsClassifier(n_neighbors=5)),
            ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000))
        ]
        
        best_accuracy = 0
        best_classifier_name = ""
        
        for name, clf in classifiers:
            try:
                print(f"  🔧 Testing {name}...")
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_classifier_name = name
                    
                print(f"    ✅ {name}: {accuracy:.3f}")
                
            except Exception as e:
                print(f"    ❌ {name} failed: {e}")
                continue
        
        accuracies.append(best_accuracy)
        fold_time = time.time() - fold_start
        print(f"  ⏱️ Fold {fold} completed in {fold_time:.1f}s")
        print(f"  🏆 Best: {best_classifier_name} ({best_accuracy:.3f})")
    
    total_time = time.time() - start_time
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n" + "=" * 60)
    print(f"📊 QUICK EVALUATION RESULTS (completed in {total_time:.1f}s)")
    print(f"=" * 60)
    print(f"🏆 Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"📈 Individual accuracies: {[f'{acc:.3f}' for acc in accuracies]}")
    print(f"=" * 60)
    
    return mean_acc, std_acc

def load_and_quick_evaluate():
    """Load saved features and run quick evaluation"""
    print("🔍 Loading saved features...")
    
    try:
        # Load features from Google Drive
        features_path = "/content/drive/MyDrive/lgmd_features.npz"
        data = np.load(features_path)
        features = data['features']
        labels = data['labels']
        
        print(f"✅ Features loaded: {features.shape}")
        print(f"✅ Labels loaded: {labels.shape}")
        
        # Run quick evaluation
        mean_acc, std_acc = quick_simple_evaluation(features, labels)
        
        return mean_acc, std_acc
        
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        print("Please ensure features are saved to Google Drive first.")
        return None, None

# For direct execution in Colab
if __name__ == "__main__":
    print("🚀 Starting Quick Evaluation...")
    mean_acc, std_acc = load_and_quick_evaluate()
    
    if mean_acc is not None:
        print(f"\n🎉 Quick evaluation completed successfully!")
        print(f"📊 Performance: {mean_acc:.4f} ± {std_acc:.4f}")
    else:
        print(f"\n❌ Quick evaluation failed!") 