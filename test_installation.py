#!/usr/bin/env python3
"""
Quick test script to verify all imports work correctly.
Run this before main.py to check your installation.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    
    print("Testing imports...")
    print("-" * 50)
    
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('pennylane', 'PennyLane'),
        ('sklearn', 'scikit-learn'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
        ('psutil', 'psutil'),
    ]
    
    failed = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {str(e)}")
            failed.append(name)
    
    print("-" * 50)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def test_project_structure():
    """Test that all required files and directories exist."""
    
    print("\nTesting project structure...")
    print("-" * 50)
    
    import os
    
    required_files = [
        'config.yaml',
        'main.py',
        'requirements.txt',
        'models/classical/cnn.py',
        'models/classical/lstm.py',
        'models/hybrid_quantum/vqc.py',
        'models/hybrid_quantum/hybrid_model.py',
        'models/quantum_kernel/qsvm.py',
        'training/trainer_classical.py',
        'training/trainer_hybrid.py',
        'training/trainer_qsvm.py',
        'evaluation/metrics.py',
        'evaluation/plots.py',
        'evaluation/logger.py',
    ]
    
    missing = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file}")
            missing.append(file)
    
    print("-" * 50)
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    else:
        print("\n✅ All required files present!")
        return True


def test_config():
    """Test that configuration file is valid."""
    
    print("\nTesting configuration...")
    print("-" * 50)
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['random_seed', 'datasets', 'dataset_sizes', 'training', 'models']
        
        for key in required_keys:
            if key in config:
                print(f"✓ {key}")
            else:
                print(f"✗ {key}")
                return False
        
        print("-" * 50)
        print("\n✅ Configuration is valid!")
        return True
        
    except Exception as e:
        print(f"✗ Error loading config: {str(e)}")
        print("-" * 50)
        return False


def main():
    """Run all tests."""
    
    print("=" * 50)
    print("QUANTUM ML BENCHMARKING - INSTALLATION TEST")
    print("=" * 50)
    print()
    
    tests = [
        test_imports(),
        test_project_structure(),
        test_config()
    ]
    
    print("\n" + "=" * 50)
    
    if all(tests):
        print("✅ ALL TESTS PASSED!")
        print("You can now run: python main.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please fix the issues above before running main.py")
        sys.exit(1)
    
    print("=" * 50)


if __name__ == "__main__":
    main()
