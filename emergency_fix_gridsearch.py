import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def emergency_fast_evaluation(features, labels):
    """
    Emergency fast evaluation to replace stuck GridSearchCV
    """
    print("🚨 EMERGENCY FAST EVALUATION")
    print("=" * 50)
    print("⚡ Replacing stuck GridSearchCV with fast alternatives")
    print("=" * 50)
    
    # 1. Reduce feature dimensionality
    print("\n📉 Step 1: Reducing feature dimensionality...")
    if features.shape[1] > 100:
        # Use feature selection to reduce dimensionality
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Select top 100 features
        selector = SelectKBest(score_func=f_classif, k=100)
        features_reduced = selector.fit_transform(features, labels)
        
        print(f"   ✅ Reduced features from {features.shape[1]} to {features_reduced.shape[1]}")
        features = features_reduced
    else:
        print(f"   ✅ Features already manageable: {features.shape[1]} dimensions")
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Fast evaluation with simple classifiers
    print("\n🤖 Step 2: Fast classifier evaluation...")
    
    classifiers = [
        ('RandomForest', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('SVM', SVC(kernel='rbf', random_state=42)),
        ('SVM_Linear', SVC(kernel='linear', random_state=42))
    ]
    
    results = {}
    best_accuracy = 0
    best_classifier = None
    
    for name, clf in classifiers:
        print(f"   🔄 Testing {name}...")
        start_time = time.time()
        
        # Use cross-validation instead of GridSearchCV
        scores = cross_val_score(clf, X_train_scaled, y_train, cv=3, scoring='accuracy')
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Test on test set
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        elapsed_time = time.time() - start_time
        
        results[name] = {
            'cv_mean': mean_score,
            'cv_std': std_score,
            'test_accuracy': test_accuracy,
            'time': elapsed_time
        }
        
        print(f"      ✅ {name}: CV={mean_score:.3f}±{std_score:.3f}, Test={test_accuracy:.3f}, Time={elapsed_time:.1f}s")
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_classifier = name
    
    # 5. Results summary
    print(f"\n🏆 RESULTS SUMMARY")
    print("=" * 50)
    print(f"Best classifier: {best_classifier}")
    print(f"Best accuracy: {best_accuracy:.3f}")
    print(f"Total evaluation time: {sum(r['time'] for r in results.values()):.1f} seconds")
    
    return results, best_classifier, best_accuracy

def quick_feature_analysis(features, labels):
    """
    Quick analysis of feature quality
    """
    print("\n📊 QUICK FEATURE ANALYSIS")
    print("=" * 40)
    
    print(f"Feature shape: {features.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Check feature variance
    feature_variance = np.var(features, axis=0)
    print(f"Feature variance range: {np.min(feature_variance):.6f} to {np.max(feature_variance):.6f}")
    
    # Check for constant features
    constant_features = np.sum(feature_variance == 0)
    print(f"Constant features: {constant_features}")
    
    # Check for NaN values
    nan_features = np.sum(np.isnan(features))
    print(f"NaN values: {nan_features}")
    
    return {
        'shape': features.shape,
        'n_classes': len(np.unique(labels)),
        'class_distribution': np.bincount(labels),
        'feature_variance_range': (np.min(feature_variance), np.max(feature_variance)),
        'constant_features': constant_features,
        'nan_values': nan_features
    }

def load_and_fix_features():
    """
    Load features and apply emergency fix
    """
    print("🔧 LOADING AND FIXING FEATURES")
    print("=" * 50)
    
    # Try to load existing features
    try:
        # Load from npz file
        data = np.load('/content/drive/MyDrive/lgmd_features.npz')
        features = data['features']
        labels = data['labels']
        print(f"✅ Loaded features: {features.shape}")
        
        # Quick analysis
        analysis = quick_feature_analysis(features, labels)
        
        # Apply emergency fix
        results, best_classifier, best_accuracy = emergency_fast_evaluation(features, labels)
        
        return features, labels, results, best_classifier, best_accuracy
        
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        print("Please ensure lgmd_features.npz exists in Google Drive")
        return None, None, None, None, None

def save_emergency_results(results, best_classifier, best_accuracy):
    """
    Save emergency evaluation results
    """
    import json
    
    emergency_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'best_classifier': best_classifier,
        'best_accuracy': best_accuracy,
        'all_results': results,
        'note': 'Emergency evaluation due to GridSearchCV timeout'
    }
    
    # Save to Google Drive
    with open('/content/drive/MyDrive/emergency_evaluation_results.json', 'w') as f:
        json.dump(emergency_results, f, indent=2)
    
    print(f"✅ Emergency results saved to Google Drive")

# Main execution function
def main_emergency_fix():
    """
    Main function to run emergency fix
    """
    print("🚨 EMERGENCY GRIDSEARCH FIX")
    print("=" * 60)
    print("⚡ This will replace the stuck GridSearchCV with fast alternatives")
    print("=" * 60)
    
    # Load and fix features
    features, labels, results, best_classifier, best_accuracy = load_and_fix_features()
    
    if features is not None:
        # Save results
        save_emergency_results(results, best_classifier, best_accuracy)
        
        print(f"\n🎉 EMERGENCY FIX COMPLETED!")
        print(f"Best accuracy: {best_accuracy:.3f}")
        print(f"Best classifier: {best_classifier}")
        
        return features, labels, results
    else:
        print("❌ Failed to load features. Please check your setup.")
        return None, None, None

# Quick usage in Colab
if __name__ == "__main__":
    # Run emergency fix
    features, labels, results = main_emergency_fix() 