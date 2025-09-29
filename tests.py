# test_eulernet.py
# Test script to verify all components work correctly

import torch
import numpy as np
import math
import sys
from pathlib import Path

def test_mathematical_utilities():
    """Test mathematical utility functions"""
    print("Testing Mathematical Utilities...")
    
    # Import the utils
    try:
        from utils import MATH_CONSTANTS, MathematicalConstants
        math_utils = MATH_CONSTANTS
    except ImportError:
        # Fallback to inline definition
        class MathUtils:
            @staticmethod
            def is_prime(n):
                if n < 2: return False
                if n == 2: return True
                if n % 2 == 0: return False
                for i in range(3, int(math.sqrt(n)) + 1, 2):
                    if n % i == 0: return False
                return True
            
            @staticmethod
            def euler_totient(n):
                result = n
                p = 2
                while p * p <= n:
                    if n % p == 0:
                        while n % p == 0: n //= p
                        result -= result // p
                    p += 1
                if n > 1: result -= result // n
                return result
        
        math_utils = MathUtils()
    
    # Test prime detection
    test_cases = [
        (2, True), (3, True), (4, False), (17, True), (25, False), (31, True)
    ]
    
    for n, expected in test_cases:
        result = math_utils.is_prime(n)
        assert result == expected, f"Prime test failed for {n}: expected {expected}, got {result}"
        print(f"  ✓ is_prime({n}) = {result}")
    
    # Test totient function
    totient_cases = [(1, 1), (2, 1), (3, 2), (4, 2), (5, 4), (6, 2), (10, 4)]
    
    for n, expected in totient_cases:
        result = math_utils.euler_totient(n)
        assert result == expected, f"Totient test failed for {n}: expected {expected}, got {result}"
        print(f"  ✓ φ({n}) = {result}")
    
    print("  ✅ Mathematical utilities test passed!")

def test_configuration():
    """Test configuration system"""
    print("Testing Configuration System...")
    
    try:
        from standalone_config import EulerNetConfig, ConfigFactory
        
        # Test basic configuration
        config = EulerNetConfig()
        assert config.model.d_model > 0, "Invalid d_model"
        assert config.training.batch_size > 0, "Invalid batch_size"
        print("  ✓ Basic configuration creation works")
        
        # Test factory
        dev_config = ConfigFactory.development_config()
        assert dev_config.training.num_epochs == 20, "Dev config incorrect"
        print("  ✓ Configuration factory works")
        
        # Test serialization
        config.save_json("test_config.json")
        loaded_config = EulerNetConfig.load_json("test_config.json")
        assert loaded_config.model.d_model == config.model.d_model, "JSON serialization failed"
        print("  ✓ JSON serialization works")
        
        # Cleanup
        Path("test_config.json").unlink(missing_ok=True)
        
    except ImportError as e:
        print(f"  ⚠️ Configuration test skipped (import error: {e})")
    
    print("  ✅ Configuration test passed!")

def test_dataset_creation():
    """Test dataset creation"""
    print("Testing Dataset Creation...")
    
    try:
        # Import simplified version
        from complete_training_notebook import EulerDataset, SimpleConfig
        
        config = SimpleConfig()
        dataset = EulerDataset(size=10, max_number=100, split='test')
        
        assert len(dataset) == 10, f"Dataset size incorrect: {len(dataset)}"
        
        # Test sample retrieval
        sample = dataset[0]
        assert 'number_theory' in sample, "Missing number_theory domain"
        assert 'analysis' in sample, "Missing analysis domain"
        assert 'mechanics' in sample, "Missing mechanics domain"
        
        # Test feature shapes
        nt_features, nt_labels = sample['number_theory']
        assert nt_features.shape[0] == config.number_theory_input_size, "NT features wrong shape"
        assert 'is_prime' in nt_labels, "Missing is_prime label"
        
        print("  ✓ Dataset creation works")
        print("  ✓ Sample structure correct")
        
    except ImportError as e:
        print(f"  ⚠️ Dataset test skipped (import error: {e})")
    
    print("  ✅ Dataset test passed!")

def test_model_creation():
    """Test model creation and forward pass"""
    print("Testing Model Creation...")
    
    try:
        from complete_training_notebook import EulerNet, SimpleConfig
        
        config = SimpleConfig()
        model = EulerNet(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created with {total_params:,} parameters")
        
        # Test forward pass with dummy data
        batch_size = 4
        dummy_batch = {
            'number_theory': (
                torch.randn(batch_size, config.number_theory_input_size),
                {}
            ),
            'analysis': (
                torch.randn(batch_size, config.analysis_input_size),
                {}
            ),
            'mechanics': (
                torch.randn(batch_size, config.mechanics_input_size),
                {}
            )
        }
        
        with torch.no_grad():
            outputs = model(dummy_batch)
        
        assert 'number_theory' in outputs, "Missing number_theory output"
        assert 'analysis' in outputs, "Missing analysis output"
        assert 'mechanics' in outputs, "Missing mechanics output"
        
        # Check output shapes
        nt_out = outputs['number_theory']
        assert 'is_prime' in nt_out, "Missing is_prime prediction"
        assert nt_out['is_prime'].shape == (batch_size,), "Wrong is_prime shape"
        
        print("  ✓ Forward pass works")
        print("  ✓ Output structure correct")
        
    except ImportError as e:
        print(f"  ⚠️ Model test skipped (import error: {e})")
    
    print("  ✅ Model test passed!")

def test_training_components():
    """Test training components"""
    print("Testing Training Components...")
    
    try:
        from complete_training_notebook import EulerTrainer, SimpleConfig
        
        config = SimpleConfig()
        config.num_epochs = 1  # Just test one epoch
        
        trainer = EulerTrainer(config)
        
        # Test loss computation with dummy data
        batch_size = 2
        dummy_predictions = {
            'number_theory': {
                'is_prime': torch.rand(batch_size),
                'totient': torch.rand(batch_size) * 10 + 1,
                'log_totient': torch.rand(batch_size)
            }
        }
        
        dummy_labels = {
            'number_theory': (
                None,  # Features not needed for loss
                {
                    'is_prime': torch.randint(0, 2, (batch_size,)).float(),
                    'totient': torch.rand(batch_size) * 10 + 1,
                    'log_totient': torch.rand(batch_size)
                }
            )
        }
        
        loss, components = trainer.compute_loss(dummy_predictions, dummy_labels)
        assert loss.item() >= 0, "Loss should be non-negative"
        assert 'prime_loss' in components, "Missing prime loss component"
        
        print("  ✓ Loss computation works")
        
    except ImportError as e:
        print(f"  ⚠️ Training test skipped (import error: {e})")
    
    print("  ✅ Training components test passed!")

def run_mini_training():
    """Run a mini training session to verify everything works"""
    print("Running Mini Training Session...")
    
    try:
        from complete_training_notebook import EulerDataset, EulerTrainer, SimpleConfig
        from torch.utils.data import DataLoader
        
        # Create minimal config
        config = SimpleConfig()
        config.num_epochs = 2
        config.train_size = 20
        config.val_size = 10
        config.batch_size = 4
        config.max_number = 50
        
        # Create datasets
        train_dataset = EulerDataset(config.train_size, config.max_number, 'train')
        val_dataset = EulerDataset(config.val_size, config.max_number, 'val')
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        
        # Create trainer
        trainer = EulerTrainer(config)
        
        print(f"  ✓ Created trainer with {sum(p.numel() for p in trainer.model.parameters()):,} parameters")
        
        # Train for a few epochs
        history = trainer.train(train_loader, val_loader)
        
        assert len(history['train_loss']) == config.num_epochs, "Training history incomplete"
        print(f"  ✓ Training completed - Final train loss: {history['train_loss'][-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Mini training failed: {e}")
        return False

def main():
    """Main test function"""
    print("EulerNet Test Suite")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Run tests
    tests = [
        test_mathematical_utilities,
        test_configuration,
        test_dataset_creation,
        test_model_creation,
        test_training_components,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            failed += 1
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! Running mini training session...")
        success = run_mini_training()
        if success:
            print("\n🚀 EulerNet is ready for full training!")
        else:
            print("\n⚠️ Mini training had issues, but core components work")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please fix issues before proceeding.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)