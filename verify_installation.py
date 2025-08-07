#!/usr/bin/env python3
"""
Verify SIGNAL installation and environment
"""

import sys
import importlib

def check_package(package_name, min_version=None):
    """Check if a package is installed and meets version requirements"""
    try:
        module = importlib.import_module(package_name.replace('-', '_'))
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

def main():
    print("="*60)
    print("SIGNAL Environment Verification")
    print("="*60)
    
    # Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Required packages
    print("Checking required packages:")
    packages = [
        'numpy',
        'cv2',
        'sklearn',
        'scipy',
        'matplotlib',
        'seaborn',
        'joblib',
        'tqdm'
    ]
    
    all_good = True
    for pkg in packages:
        if not check_package(pkg):
            all_good = False
    
    print()
    if all_good:
        print("✅ All dependencies are installed!")
        print("You can now run: python SIGNAL_model.py")
    else:
        print("❌ Some dependencies are missing.")
        print("Please run: pip install -r requirements.txt")
    
    # Test imports
    print("\nTesting SIGNAL imports...")
    try:
        from SIGNAL_model import SIGNALModel, SIGNALConfig
        print("✅ SIGNAL model imports successfully")
    except Exception as e:
        print(f"❌ Error importing SIGNAL: {e}")
    
    print("="*60)

if __name__ == "__main__":
    main()