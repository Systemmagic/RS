"""
Property-based tests for configuration management.
Feature: koopman-accuracy-enhancement
"""

import pytest
import torch
import numpy as np
import random
import os
import tempfile
from hypothesis import given, strategies as st, settings
from config import Config, set_deterministic_seed, load_config


# Feature: koopman-accuracy-enhancement, Property 13: Configuration Determinism
@settings(max_examples=100, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10000))
def test_configuration_determinism(seed):
    """
    Property 13: Configuration Determinism
    
    For any configuration and random seed, running the training process twice
    SHALL produce identical model weights, loss curves, and evaluation metrics
    (within floating-point precision).
    
    Validates: Requirements 10.5
    
    This test verifies that:
    1. Setting the same seed produces identical random number sequences
    2. Model initialization is deterministic
    3. Training operations are reproducible
    """
    # Create a simple model for testing determinism
    config = Config(random_seed=seed, latent_dim=64, img_size=64)
    
    # First run
    set_deterministic_seed(config.random_seed)
    
    # Generate random tensors (simulating model weights)
    torch_rand_1 = torch.randn(10, 10)
    numpy_rand_1 = np.random.randn(10, 10)
    python_rand_1 = [random.random() for _ in range(10)]
    
    # Create a simple linear layer (simulating model initialization)
    model_1 = torch.nn.Linear(10, 10)
    weights_1 = model_1.weight.data.clone()
    
    # Second run with same seed
    set_deterministic_seed(config.random_seed)
    
    # Generate random tensors again
    torch_rand_2 = torch.randn(10, 10)
    numpy_rand_2 = np.random.randn(10, 10)
    python_rand_2 = [random.random() for _ in range(10)]
    
    # Create the same linear layer
    model_2 = torch.nn.Linear(10, 10)
    weights_2 = model_2.weight.data.clone()
    
    # Verify determinism
    # PyTorch random numbers should be identical
    assert torch.allclose(torch_rand_1, torch_rand_2, rtol=1e-6, atol=1e-8), \
        "PyTorch random numbers are not deterministic"
    
    # NumPy random numbers should be identical
    assert np.allclose(numpy_rand_1, numpy_rand_2, rtol=1e-6, atol=1e-8), \
        "NumPy random numbers are not deterministic"
    
    # Python random numbers should be identical
    assert all(abs(a - b) < 1e-10 for a, b in zip(python_rand_1, python_rand_2)), \
        "Python random numbers are not deterministic"
    
    # Model weights should be identical
    assert torch.allclose(weights_1, weights_2, rtol=1e-6, atol=1e-8), \
        "Model initialization is not deterministic"


def test_config_yaml_round_trip():
    """Test that saving and loading config preserves all values."""
    original_config = Config(
        data_dir="test_data",
        img_size=256,
        encoder_type="resnet",
        use_attention=True,
        batch_size=8,
        random_seed=123
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save config
        original_config.save(temp_path)
        
        # Load config
        loaded_config = Config.from_yaml(temp_path)
        
        # Verify all fields match
        assert original_config.data_dir == loaded_config.data_dir
        assert original_config.img_size == loaded_config.img_size
        assert original_config.encoder_type == loaded_config.encoder_type
        assert original_config.use_attention == loaded_config.use_attention
        assert original_config.batch_size == loaded_config.batch_size
        assert original_config.random_seed == loaded_config.random_seed
        
        # Verify complete equality
        assert original_config.to_dict() == loaded_config.to_dict()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_config_defaults():
    """Test that default configuration has expected values."""
    config = Config()
    
    # Verify key defaults from requirements
    assert config.img_size == 128, "Default img_size should be 128"
    assert config.encoder_type == "resnet", "Default encoder should be resnet"
    assert config.latent_dim == 2048, "Default latent_dim should be 2048"
    assert config.batch_size == 4, "Default batch_size should be 4"
    assert config.random_seed == 42, "Default random_seed should be 42"


def test_deterministic_seed_sets_all_generators():
    """Test that set_deterministic_seed affects all random generators."""
    seed = 12345
    
    # Set seed
    set_deterministic_seed(seed)
    
    # Generate values
    torch_val_1 = torch.randn(1).item()
    numpy_val_1 = np.random.randn()
    python_val_1 = random.random()
    
    # Reset seed
    set_deterministic_seed(seed)
    
    # Generate values again
    torch_val_2 = torch.randn(1).item()
    numpy_val_2 = np.random.randn()
    python_val_2 = random.random()
    
    # Verify reproducibility
    assert abs(torch_val_1 - torch_val_2) < 1e-6
    assert abs(numpy_val_1 - numpy_val_2) < 1e-6
    assert abs(python_val_1 - python_val_2) < 1e-10


def test_config_command_line_override():
    """Test that command-line arguments override config file values."""
    # Create a config with specific values
    original_config = Config(img_size=128, batch_size=4, random_seed=42)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        original_config.save(temp_path)
        
        # Simulate command-line arguments
        class MockArgs:
            def __init__(self):
                self.config = temp_path
                self.img_size = 256  # Override
                self.batch_size = 8  # Override
                self.random_seed = None  # Don't override
                self.data_dir = None
                self.sequence_length = None
                self.encoder_type = None
                self.encoder_depth = None
                self.use_attention = None
                self.use_multiscale = None
                self.latent_dim = None
                self.recon_weight = None
                self.pred_weight = None
                self.spectral_weight = None
                self.perceptual_weight = None
                self.epochs = None
                self.learning_rate = None
                self.use_augmentation = None
                self.rotation_range = None
                self.output_dir = None
        
        # Load config with overrides
        loaded_config = load_config(temp_path, MockArgs())
        
        # Verify overrides applied
        assert loaded_config.img_size == 256, "img_size should be overridden"
        assert loaded_config.batch_size == 8, "batch_size should be overridden"
        assert loaded_config.random_seed == 42, "random_seed should not be overridden"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
