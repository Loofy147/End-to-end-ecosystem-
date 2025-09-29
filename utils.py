# utils.py
# EulerNet Utility Functions and Helper Classes
# Common utilities used across the project

import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class MathematicalConstants:
    """Simplified mathematical constants class for standalone operation"""
    
    EULER_MASCHERONI = 0.5772156649015329
    PI = math.pi
    E = math.e
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def is_prime(n: int) -> bool:
        """Simple primality test"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    @staticmethod
    def generate_primes_sieve(limit: int) -> List[int]:
        """Simple sieve of Eratosthenes"""
        if limit < 2:
            return []
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    @lru_cache(maxsize=5000)
    def euler_totient(self, n: int) -> int:
        """Euler's totient function"""
        if n <= 1:
            return 1 if n == 1 else 0
        
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result
    
    def harmonic_number(self, n: int) -> float:
        """Compute n-th harmonic number"""
        if n <= 0:
            return 0.0
        return sum(1.0 / k for k in range(1, n + 1))
    
    def riemann_zeta_simple(self, s: float, max_terms: int = 1000) -> float:
        """Simple approximation of Riemann zeta function"""
        if s <= 1:
            return float('inf')
        
        result = 0.0
        for n in range(1, max_terms + 1):
            result += 1.0 / (n ** s)
        return result

# Global instance for easy access
MATH_CONSTANTS = MathematicalConstants()

def setup_reproducibility(seed: int = 42):
    """Setup reproducible random seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, filepath: str):
    """Save training checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint

def validate_tensor_shapes(tensors: Dict[str, torch.Tensor], 
                          expected_shapes: Dict[str, tuple]) -> bool:
    """Validate tensor shapes"""
    for name, tensor in tensors.items():
        if name in expected_shapes:
            expected = expected_shapes[name]
            actual = tensor.shape
            if len(expected) != len(actual):
                logger.error(f"Shape mismatch for {name}: expected {expected}, got {actual}")
                return False
            for i, (exp, act) in enumerate(zip(expected, actual)):
                if exp != -1 and exp != act:  # -1 means any size
                    logger.error(f"Shape mismatch for {name} dim {i}: expected {exp}, got {act}")
                    return False
    return True

def create_directory_structure():
    """Create necessary directories for the project"""
    directories = [
        'data',
        'checkpoints',
        'logs',
        'results',
        'configs',
        'plots'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure created")

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def _is_better(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

class MovingAverage:
    """Simple moving average for metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = []
        
    def update(self, value: float):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    def get_average(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0
    
    def reset(self):
        self.values = []

def format_time(seconds: float) -> str:
    """Format time duration"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def memory_usage() -> str:
    """Get current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    else:
        import psutil
        memory = psutil.virtual_memory()
        return f"RAM: {memory.used / 1024**3:.2f}GB used, {memory.available / 1024**3:.2f}GB available"

def log_system_info():
    """Log system information"""
    logger.info("System Information:")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info(f"  Device: {get_device()}")
    logger.info(f"  {memory_usage()}")

if __name__ == "__main__":
    # Test utilities
    setup_reproducibility(42)
    log_system_info()
    create_directory_structure()
    
    # Test mathematical constants
    print(f"Is 17 prime? {MATH_CONSTANTS.is_prime(17)}")
    print(f"Primes up to 20: {MATH_CONSTANTS.generate_primes_sieve(20)}")
    print(f"φ(10) = {MATH_CONSTANTS.euler_totient(10)}")
    print(f"H_5 = {MATH_CONSTANTS.harmonic_number(5)}")
    print(f"ζ(2) ≈ {MATH_CONSTANTS.riemann_zeta_simple(2.0)}")