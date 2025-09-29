# standalone_config.py
# Standalone Configuration System for EulerNet
# Self-contained configuration that doesn't depend on other modules

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
from pathlib import Path
import json
import yaml

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout_rate: float = 0.1
    activation: str = 'relu'
    
    # Domain-specific sizes
    number_theory_input_size: int = 50  # Increased for comprehensive features
    analysis_input_size: int = 12
    mechanics_input_size: int = 10
    
    # Architecture specifics
    use_attention: bool = True
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    
    # Numerical stability
    numerical_stability: bool = True
    gradient_clipping: float = 1.0
    weight_initialization: str = 'xavier_uniform'

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 100
    warmup_epochs: int = 10
    
    # Optimizer settings
    optimizer: str = 'eulerian'
    euler_momentum: float = 0.9
    weight_decay: float = 1e-4
    mathematical_regularization: float = 1e-4
    
    # Scheduler settings
    scheduler: str = 'cosine'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Loss weighting
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        'number_theory': 1.0,
        'analysis': 1.0,
        'mechanics': 1.0,
        'consistency': 0.1
    })
    
    # Curriculum learning
    curriculum_learning: bool = True
    curriculum_stages: List[str] = field(default_factory=lambda: [
        'basic_arithmetic',
        'elementary_number_theory',
        'intermediate_analysis',
        'advanced_euler_theorems',
        'unified_mathematics'
    ])
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_delta: float = 1e-4

@dataclass
class DataConfig:
    """Data configuration"""
    train_size: int = 10000
    val_size: int = 2000
    test_size: int = 1000
    
    # Data generation parameters
    max_number: int = 1000
    max_prime_limit: int = 100
    precision_digits: int = 15  # Reduced for compatibility
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_noise: float = 0.01
    
    # Data loading
    num_workers: int = 2  # Reduced for compatibility
    pin_memory: bool = True
    shuffle: bool = True

@dataclass
class SystemConfig:
    """System configuration"""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = False  # Disabled by default for compatibility
    compile_model: bool = False
    
    # Memory management
    max_memory_gb: float = 8.0
    gradient_accumulation_steps: int = 1
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = 'INFO'
    log_file: Optional[str] = 'eulernet.log'
    console_logging: bool = True
    
    # Experiment tracking (disabled by default for standalone operation)
    use_wandb: bool = False
    wandb_project: str = 'eulernet'
    wandb_entity: Optional[str] = None
    
    # Tensorboard
    use_tensorboard: bool = False
    tensorboard_log_dir: str = 'runs'
    
    # Model checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_frequency: int = 10
    keep_last_n_checkpoints: int = 3
    
    # Metrics logging
    log_frequency: int = 100
    eval_frequency: int = 500

@dataclass
class EulerNetConfig:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Metadata
    version: str = '1.0.0'
    experiment_name: str = 'eulernet_experiment'
    description: str = 'EulerNet: Neural Network Implementation of Euler\'s Mathematical Universe'
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        self._setup_directories()
    
    def _validate_config(self):
        """Basic validation"""
        # Model validation
        assert self.model.d_model > 0, "d_model must be positive"
        assert self.model.n_heads > 0, "n_heads must be positive"
        assert self.model.d_model % self.model.n_heads == 0, "d_model must be divisible by n_heads"
        
        # Training validation
        assert self.training.batch_size > 0, "batch_size must be positive"
        assert 0 < self.training.learning_rate < 1, "learning_rate must be between 0 and 1"
        assert self.training.num_epochs > 0, "num_epochs must be positive"
        
        # Data validation
        assert self.data.train_size > 0, "train_size must be positive"
        assert self.data.val_size > 0, "val_size must be positive"
        assert self.data.test_size > 0, "test_size must be positive"
        
        # System validation
        if self.system.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, switching to CPU")
            self.system.device = 'cpu'
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.logging.checkpoint_dir,
            self.logging.tensorboard_log_dir,
            'logs',
            'results',
            'data'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'system': self.system.__dict__,
            'logging': self.logging.__dict__,
            'version': self.version,
            'experiment_name': self.experiment_name,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'EulerNetConfig':
        """Create from dictionary"""
        config = cls()
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        if 'system' in config_dict:
            for key, value in config_dict['system'].items():
                if hasattr(config.system, key):
                    setattr(config.system, key, value)
        
        if 'logging' in config_dict:
            for key, value in config_dict['logging'].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
        
        for key in ['version', 'experiment_name', 'description']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def save_json(self, filepath: str):
        """Save to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_yaml(self, filepath: str):
        """Save to YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'EulerNetConfig':
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def load_yaml(cls, filepath: str) -> 'EulerNetConfig':
        """Load from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

class ConfigFactory:
    """Factory for creating different configuration presets"""
    
    @staticmethod
    def minimal_config() -> EulerNetConfig:
        """Minimal configuration for testing"""
        config = EulerNetConfig()
        config.training.num_epochs = 5
        config.training.batch_size = 16
        config.data.train_size = 100
        config.data.val_size = 20
        config.data.test_size = 10
        config.data.max_number = 100
        config.model.d_model = 128
        config.model.n_heads = 4
        return config
    
    @staticmethod
    def development_config() -> EulerNetConfig:
        """Development configuration"""
        config = EulerNetConfig()
        config.training.num_epochs = 20
        config.training.batch_size = 32
        config.data.train_size = 1000
        config.data.val_size = 200
        config.data.test_size = 100
        config.data.max_number = 500
        return config
    
    @staticmethod
    def production_config() -> EulerNetConfig:
        """Production configuration"""
        config = EulerNetConfig()
        config.training.num_epochs = 200
        config.training.batch_size = 128
        config.data.train_size = 50000
        config.data.val_size = 10000
        config.data.test_size = 5000
        config.data.max_number = 10000
        config.system.mixed_precision = True
        config.system.compile_model = True
        return config

# Test the configuration system
if __name__ == "__main__":
    print("Testing EulerNet Configuration System...")
    
    # Create different configurations
    minimal = ConfigFactory.minimal_config()
    dev = ConfigFactory.development_config()
    prod = ConfigFactory.production_config()
    
    print("✅ Configuration creation successful")
    
    # Test serialization
    dev.save_json("test_config.json")
    dev.save_yaml("test_config.yaml")
    
    # Test loading
    loaded_json = EulerNetConfig.load_json("test_config.json")
    loaded_yaml = EulerNetConfig.load_yaml("test_config.yaml")
    
    print("✅ Configuration serialization/loading successful")
    
    # Test validation
    try:
        invalid_config = EulerNetConfig()
        invalid_config.model.d_model = -1  # Should fail validation
        invalid_config._validate_config()
    except AssertionError:
        print("✅ Configuration validation working correctly")
    
    print("Configuration system test completed!")
    
    # Clean up test files
    import os
    for file in ["test_config.json", "test_config.yaml"]:
        if os.path.exists(file):
            os.remove(file)