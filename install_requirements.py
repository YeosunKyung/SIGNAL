#!/usr/bin/env python3
"""
Requirements installation script for LGMD Project
"""

import subprocess
import sys
import importlib

def check_and_install_package(package_name, install_name=None):
    """Check if package is installed, install if not"""
    if install_name is None:
        install_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False

def main():
    print("=" * 50)
    print("LGMD Project Requirements Installation")
    print("=" * 50)
    
    # Core scientific computing packages
    packages = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("cv2", "opencv-python"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy"),
        ("geoopt", "geoopt"),
    ]
    
    all_installed = True
    
    for package_name, install_name in packages:
        if not check_and_install_package(package_name, install_name):
            all_installed = False
    
    print("\n" + "=" * 50)
    if all_installed:
        print("✅ All required packages are installed!")
        print("You can now run the LGMD encoder code.")
    else:
        print("❌ Some packages failed to install.")
        print("Please install them manually or check your internet connection.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 