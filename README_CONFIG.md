# Configuration Management Guide

## Overview

The configuration management system provides a centralized way to manage all hyperparameters for the Deep Koopman PM2.5 forecasting model. It supports YAML configuration files and command-line argument overrides.

## Files

- `config.py`: Configuration dataclass and utilities
- `config.yaml`: Default configuration file with all hyperparameters
- `test_config.py`: Property-based tests for configuration determinism

## Usage

### 1. Using Default Configuration

```python
from config import Config, set_deterministic_seed

# Load default configuration
config = Config()

# Set deterministic seed for reproducibility
set_deterministic_seed(config.random_seed)
```

### 2. Loading from YAML File

```python
from config import Config

# Load configuration from YAML
config = Config.from_yaml('config.yaml')

# Modify and save
config.img_size = 256
config.save('config_256.yaml')
```

### 3. Command-Line Arguments

```bash
# Use default config
python train.py

# Override specific parameters
python train.py --img_size 256 --batch_size 2 --epochs 100

# Use custom config file
python train.py --config my_config.yaml --learning_rate 0.0001
```

### 4. Programmatic Override

```python
from config import Config, load_config, parse_args

# Parse command-line arguments
args = parse_args()

# Load config with overrides
config = load_config(args.config, args)
```

## Configuration Parameters

### Data Parameters
- `data_dir`: Directory containing PM2.5 GeoTIFF files
- `img_size`: Target image size (128 or 256)
- `sequence_length`: Number of frames in temporal sequence

### Model Architecture
- `encoder_type`: "simple" or "resnet"
- `encoder_depth`: Number of residual blocks (for ResNet)
- `use_attention`: Enable spatial attention mechanisms
- `use_multiscale`: Enable multi-scale feature extraction
- `latent_dim`: Dimension of latent space

### Loss Weights
- `recon_weight`: Reconstruction loss weight
- `pred_weight`: Prediction loss weight
- `spectral_weight`: Spectral regularization weight
- `perceptual_weight`: Perceptual loss weight
- `linearity_weight`: Linearity constraint weight
- `time_decay_gamma`: Temporal decay factor

### Training Parameters
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `learning_rate`: Optimizer learning rate

### Augmentation
- `use_augmentation`: Enable data augmentation
- `rotation_range`: Maximum rotation angle (degrees)

### Evaluation
- `forecast_days`: Number of days to forecast
- `start_day_idx`: Starting day index for evaluation

### Output
- `output_dir`: Directory for results
- `checkpoint_interval`: Save checkpoint every N epochs

### Reproducibility
- `random_seed`: Random seed for deterministic behavior

## Deterministic Training

The configuration system ensures reproducible results:

```python
from config import Config, set_deterministic_seed

config = Config(random_seed=42)
set_deterministic_seed(config.random_seed)

# All random operations will be deterministic:
# - PyTorch random number generation
# - NumPy random number generation
# - Python random module
# - CUDA operations (when deterministic=True)
```

## Testing

Run property-based tests to verify configuration determinism:

```bash
# Run all configuration tests
python -m pytest test_config.py -v

# Run only the property-based determinism test
python -m pytest test_config.py::test_configuration_determinism -v
```

The property test verifies that:
1. Setting the same seed produces identical random sequences
2. Model initialization is deterministic
3. Training operations are reproducible

## Example: Training with Custom Configuration

```python
from config import Config, set_deterministic_seed, load_config, parse_args

# Parse arguments
args = parse_args()

# Load configuration
config = load_config(args.config, args)

# Set deterministic seed
set_deterministic_seed(config.random_seed)

# Save configuration for this experiment
config.save(f"{config.output_dir}/config_used.yaml")

# Use configuration in training
print(f"Training with img_size={config.img_size}, batch_size={config.batch_size}")
# ... training code ...
```

## Requirements Validation

This implementation satisfies:
- **Requirement 10.1**: Configuration loaded from YAML/JSON files ✓
- **Requirement 10.2**: Command-line arguments override config parameters ✓
- **Requirement 10.5**: Deterministic behavior with same seed and config ✓

The property-based test `test_configuration_determinism` validates Requirement 10.5 by running 100 iterations with different random seeds and verifying that all random number generators produce identical sequences when reseeded.
