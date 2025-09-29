# datasets.py
# EulerNet Data Generation and Dataset Management
# Professional data pipeline for mathematical neural networks

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pickle
import logging
from pathlib import Path
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Remove the problematic imports for now
# from .mathematical_constants import EULER_CONSTANTS, DATA_FACTORY
# from .config import EulerNetConfig

# We'll create local instances instead
import sys
from pathlib import Path

# Add the current directory to sys.path to allow imports
current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class DataSample:
    """Standardized data sample structure"""
    features: torch.Tensor
    labels: Dict[str, torch.Tensor]
    metadata: Dict[str, Any] = None
    sample_id: str = None
    difficulty_level: float = 0.0

class BaseEulerDataset(Dataset, ABC):
    """Abstract base class for all Euler datasets"""
    
    def __init__(self, config: EulerNetConfig, split: str = 'train'):
        self.config = config
        self.split = split
        self.data_samples = []
        self.scaler = None
        self.difficulty_weights = None
        
        # Initialize based on split
        self._initialize_dataset()
        
    @abstractmethod
    def _generate_sample(self, index: int) -> DataSample:
        """Generate a single data sample"""
        pass
    
    @abstractmethod
    def _initialize_dataset(self):
        """Initialize the dataset for the given split"""
        pass
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        return self.data_samples[idx]
    
    def get_feature_stats(self) -> Dict[str, torch.Tensor]:
        """Compute feature statistics for normalization"""
        if not self.data_samples:
            return {}
        
        all_features = torch.stack([sample.features for sample in self.data_samples])
        return {
            'mean': torch.mean(all_features, dim=0),
            'std': torch.std(all_features, dim=0),
            'min': torch.min(all_features, dim=0)[0],
            'max': torch.max(all_features, dim=0)[0]
        }

class NumberTheoryDataset(BaseEulerDataset):
    """Dataset for number theory problems (primes, totient function, etc.)"""
    
    def _initialize_dataset(self):
        """Initialize number theory dataset"""
        size_map = {
            'train': self.config.data.train_size,
            'val': self.config.data.val_size,
            'test': self.config.data.test_size
        }
        
        dataset_size = size_map[self.split]
        self.data_samples = [self._generate_sample(i) for i in range(dataset_size)]
        
        if self.split == 'train' and self.config.data.use_augmentation:
            self._apply_augmentation()
    
    def _generate_sample(self, index: int) -> DataSample:
        """Generate a number theory sample"""
        # Random integer in range
        n = random.randint(2, self.config.data.max_number)
        
        # Generate comprehensive features
        features = self._extract_number_features(n)
        
        # Generate labels
        labels = self._compute_number_labels(n)
        
        # Compute difficulty based on number properties
        difficulty = self._compute_difficulty(n)
        
        return DataSample(
            features=features,
            labels=labels,
            metadata={'original_number': n},
            sample_id=f"nt_{self.split}_{index}",
            difficulty_level=difficulty
        )
    
    def _extract_number_features(self, n: int) -> torch.Tensor:
        """Extract comprehensive features from a number"""
        from .utils import MATH_CONSTANTS
        
        features = []
        
        # Basic number properties
        features.append(float(n))
        features.append(float(np.log(n)))
        features.append(float(np.sqrt(n)))
        
        # Binary representation (fixed width)
        binary_repr = self._to_binary_features(n, max_bits=16)
        features.extend(binary_repr)
        
        # Digit-based features
        digit_str = str(n)
        features.append(len(digit_str))  # Number of digits
        features.append(sum(int(d) for d in digit_str))  # Digit sum
        features.append(sum(int(d)**2 for d in digit_str))  # Sum of squares
        
        # Divisibility features
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19]
        for p in small_primes:
            features.append(float(n % p == 0))  # Exact divisibility
            features.append(float(n % p))  # Remainder
        
        # Mathematical properties
        features.append(float(n % 2 == 0))  # Even/odd
        features.append(float(n % 4))  # Quadratic residue class mod 4
        features.append(float(n % 8))  # Residue class mod 8
        
        # Number-theoretic functions (approximate)
        features.append(float(len([d for d in range(1, min(n+1, 100)) if n % d == 0])))  # Divisor count (limited)
        features.append(float(sum(d for d in range(1, min(n+1, 100)) if n % d == 0)))  # Divisor sum (limited)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _to_binary_features(self, n: int, max_bits: int = 16) -> List[float]:
        """Convert number to fixed-width binary features"""
        binary_str = format(n, f'0{max_bits}b')[-max_bits:]  # Take last max_bits
        return [float(bit) for bit in binary_str]
    
    def _compute_number_labels(self, n: int) -> Dict[str, torch.Tensor]:
        """Compute all labels for a number"""
        from .utils import MATH_CONSTANTS
        
        labels = {}
        
        # Primary classification: is prime?
        labels['is_prime'] = torch.tensor(float(MATH_CONSTANTS.is_prime(n)))
        
        # Euler's totient function
        labels['totient'] = torch.tensor(float(MATH_CONSTANTS.euler_totient(n)))
        
        # Prime factorization (encoded as vector)
        prime_factors = self._encode_prime_factorization(n)
        labels['prime_factors'] = torch.tensor(prime_factors, dtype=torch.float32)
        
        # Legendre symbols for small primes
        legendre_symbols = self._compute_legendre_symbols(n)
        labels['legendre_symbols'] = torch.tensor(legendre_symbols, dtype=torch.float32)
        
        # Möbius function (simplified)
        labels['mobius'] = torch.tensor(float(self._mobius_simple(n)))
        
        # Additional number-theoretic functions
        labels['is_perfect_power'] = torch.tensor(float(self._is_perfect_power(n)))
        labels['is_square_free'] = torch.tensor(float(abs(self._mobius_simple(n)) == 1))
        
        return labels
    
    def _encode_prime_factorization(self, n: int, max_factors: int = 20) -> List[float]:
        """Encode prime factorization as fixed-length vector"""
        from .utils import MATH_CONSTANTS
        
        factors = []
        temp_n = n
        
        # Find all prime factors with multiplicities
        primes = MATH_CONSTANTS.generate_primes_sieve(min(1000, int(np.sqrt(n)) + 1))
        for p in primes:
            count = 0
            while temp_n % p == 0:
                temp_n //= p
                count += 1
            if count > 0:
                factors.extend([p, count])  # [prime, multiplicity]
        
        # Handle remaining prime factor
        if temp_n > 1:
            factors.extend([temp_n, 1])
        
        # Pad or truncate to fixed length
        while len(factors) < max_factors:
            factors.append(0)
        
        return factors[:max_factors]
    
    def _compute_legendre_symbols(self, n: int) -> List[float]:
        """Compute Legendre symbols for small primes (simplified)"""
        test_primes = [3, 5, 7, 11, 13, 17, 19, 23]
        symbols = []
        
        for p in test_primes:
            if n % p == 0:
                symbols.append(0.0)
            else:
                # Simplified Legendre symbol computation
                symbol = pow(n, (p - 1) // 2, p)
                symbols.append(1.0 if symbol == 1 else -1.0)
        
        return symbols
    
    def _mobius_simple(self, n: int) -> int:
        """Simplified Möbius function"""
        from .utils import MATH_CONSTANTS
        
        if n <= 0:
            return 0
        if n == 1:
            return 1
        
        # Simple implementation
        factors = []
        temp_n = n
        d = 2
        
        while d * d <= temp_n:
            count = 0
            while temp_n % d == 0:
                temp_n //= d
                count += 1
            
            if count > 0:
                if count > 1:  # Square factor found
                    return 0
                factors.append(d)
            
            d += 1
        
        if temp_n > 1:
            factors.append(temp_n)
        
        # Return (-1)^k where k is number of prime factors
        return (-1) ** len(factors)
    
    def _is_perfect_power(self, n: int) -> bool:
        """Check if n is a perfect power (k-th power for some k > 1)"""
        if n <= 1:
            return n == 1
        
        for k in range(2, int(np.log2(n)) + 1):
            root = round(n ** (1.0 / k))
            if root ** k == n:
                return True
        
        return False
    
    def _compute_difficulty(self, n: int) -> float:
        """Compute sample difficulty for curriculum learning"""
        from .utils import MATH_CONSTANTS
        
        difficulty = 0.0
        
        # Size-based difficulty
        difficulty += np.log10(n) / 4.0  # Larger numbers are harder
        
        # Prime-based difficulty
        if MATH_CONSTANTS.is_prime(n):
            difficulty += 0.3  # Primes are harder to classify
        
        # Compositeness complexity
        primes = MATH_CONSTANTS.generate_primes_sieve(min(100, int(np.sqrt(n)) + 1))
        num_prime_factors = len([p for p in primes if n % p == 0 and MATH_CONSTANTS.is_prime(p)])
        difficulty += num_prime_factors * 0.1
        
        # Special number types
        if self._is_perfect_power(n):
            difficulty += 0.2
        
        return min(difficulty, 1.0)  # Cap at 1.0
    
    def _apply_augmentation(self):
        """Apply data augmentation for training set"""
        if not self.config.data.use_augmentation:
            return
        
        original_samples = self.data_samples.copy()
        augmented_samples = []
        
        for sample in original_samples:
            # Feature noise augmentation
            if random.random() < 0.3:  # 30% chance
                noisy_features = sample.features + torch.randn_like(sample.features) * self.config.data.augmentation_noise
                
                augmented_sample = DataSample(
                    features=noisy_features,
                    labels=sample.labels.copy(),
                    metadata=sample.metadata.copy(),
                    sample_id=sample.sample_id + "_aug",
                    difficulty_level=sample.difficulty_level
                )
                augmented_samples.append(augmented_sample)
        
        self.data_samples.extend(augmented_samples)
        logger.info(f"Applied augmentation: {len(original_samples)} -> {len(self.data_samples)} samples")

class AnalysisDataset(BaseEulerDataset):
    """Dataset for mathematical analysis problems (zeta function, series, etc.)"""
    
    def _initialize_dataset(self):
        """Initialize analysis dataset"""
        size_map = {
            'train': self.config.data.train_size // 2,  # Smaller for computationally intensive
            'val': self.config.data.val_size // 2,
            'test': self.config.data.test_size // 2
        }
        
        dataset_size = size_map[self.split]
        self.data_samples = [self._generate_sample(i) for i in range(dataset_size)]
    
    def _generate_sample(self, index: int) -> DataSample:
        """Generate an analysis sample"""
        # Random s value for zeta function (s > 1 for convergence)
        s = random.uniform(1.1, 10.0)
        
        # Random n for harmonic numbers
        n = random.randint(1, 10000)
        
        features = self._extract_analysis_features(s, n)
        labels = self._compute_analysis_labels(s, n)
        difficulty = self._compute_analysis_difficulty(s, n)
        
        return DataSample(
            features=features,
            labels=labels,
            metadata={'s_value': s, 'n_value': n},
            sample_id=f"analysis_{self.split}_{index}",
            difficulty_level=difficulty
        )
    
    def _extract_analysis_features(self, s: float, n: int) -> torch.Tensor:
        """Extract features for analysis problems"""
        features = []
        
        # Zeta function parameter
        features.append(s)
        features.append(s - 1)  # Distance from pole
        features.append(1.0 / s)
        features.append(np.log(s))
        
        # Harmonic series parameter
        features.append(float(n))
        features.append(np.log(n))
        features.append(1.0 / n)
        features.append(np.sqrt(n))
        
        # Partial computations
        harmonic_partial = sum(1.0 / k for k in range(1, min(n, 1000) + 1))
        features.append(harmonic_partial)
        
        # Euler product approximation (small number of terms)
        euler_product_partial = 1.0
        primes = EULER_CONSTANTS.generate_primes_sieve(100)[:20]  # First 20 primes
        for p in primes:
            euler_product_partial *= 1.0 / (1.0 - p**(-s))
        features.append(euler_product_partial)
        
        # Series convergence indicators
        features.append(s - 1.0)  # Convergence margin for zeta
        features.append(1.0 / (s - 1.0) if s > 1.01 else 100.0)  # Pole proximity
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _compute_analysis_labels(self, s: float, n: int) -> Dict[str, torch.Tensor]:
        """Compute labels for analysis problems"""
        from .utils import MATH_CONSTANTS
        
        labels = {}
        
        # Riemann zeta function value (simplified)
        labels['zeta_value'] = torch.tensor(float(MATH_CONSTANTS.riemann_zeta_simple(s)))
        
        # Harmonic number
        labels['harmonic_number'] = torch.tensor(float(MATH_CONSTANTS.harmonic_number(n)))
        
        # Euler-Mascheroni constant approximation
        harmonic = MATH_CONSTANTS.harmonic_number(n)
        gamma_approx = harmonic - np.log(n) if n > 0 else MATH_CONSTANTS.EULER_MASCHERONI
        labels['gamma_approximation'] = torch.tensor(float(gamma_approx))
        
        # Euler product (simplified approximation)
        euler_product = self._simple_euler_product(s, 20)
        labels['euler_product'] = torch.tensor(float(euler_product))
        
        # Series convergence properties
        series_sum = sum(1.0 / (k**s) for k in range(1, min(1000, n) + 1))
        labels['series_sum_partial'] = torch.tensor(float(series_sum))
        
        return labels
    
    def _simple_euler_product(self, s: float, max_primes: int = 20) -> float:
        """Simple Euler product approximation"""
        from .utils import MATH_CONSTANTS
        
        if s <= 1:
            return float('inf')
        
        primes = MATH_CONSTANTS.generate_primes_sieve(100)[:max_primes]
        product = 1.0
        
        for p in primes:
            try:
                factor = 1.0 / (1.0 - p**(-s))
                product *= factor
                if product > 1e10:  # Prevent overflow
                    break
            except (OverflowError, ZeroDivisionError):
                break
        
        return product
    
    def _compute_analysis_difficulty(self, s: float, n: int) -> float:
        """Compute difficulty for analysis problems"""
        difficulty = 0.0
        
        # Proximity to zeta function pole
        pole_distance = abs(s - 1.0)
        difficulty += max(0.0, 0.5 - pole_distance)  # Harder near s=1
        
        # Large n values are harder for harmonic series
        difficulty += min(0.3, np.log10(n) / 10.0)
        
        # Special values are easier
        if abs(s - round(s)) < 0.01 and s > 1:
            difficulty -= 0.1
        
        return max(0.0, min(1.0, difficulty))

class MechanicsDataset(BaseEulerDataset):
    """Dataset for Lagrangian mechanics problems"""
    
    def _initialize_dataset(self):
        """Initialize mechanics dataset"""
        size_map = {
            'train': self.config.data.train_size // 3,
            'val': self.config.data.val_size // 3,
            'test': self.config.data.test_size // 3
        }
        
        dataset_size = size_map[self.split]
        self.data_samples = [self._generate_sample(i) for i in range(dataset_size)]
    
    def _generate_sample(self, index: int) -> DataSample:
        """Generate a mechanics sample"""
        # System parameters
        system_type = random.choice(['harmonic_oscillator', 'pendulum', 'particle_in_potential'])
        
        if system_type == 'harmonic_oscillator':
            features, labels = self._generate_harmonic_oscillator()
        elif system_type == 'pendulum':
            features, labels = self._generate_pendulum()
        else:
            features, labels = self._generate_particle_in_potential()
        
        difficulty = self._compute_mechanics_difficulty(system_type, features)
        
        return DataSample(
            features=features,
            labels=labels,
            metadata={'system_type': system_type},
            sample_id=f"mechanics_{system_type}_{self.split}_{index}",
            difficulty_level=difficulty
        )
    
    def _generate_harmonic_oscillator(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate harmonic oscillator problem"""
        # System parameters
        mass = random.uniform(0.5, 3.0)
        spring_k = random.uniform(1.0, 20.0)
        
        # Initial conditions
        q0 = random.uniform(-3.0, 3.0)
        q_dot0 = random.uniform(-3.0, 3.0)
        
        # Time points
        t = random.uniform(0, 4 * np.pi)
        
        # Features
        features = torch.tensor([
            mass, spring_k, q0, q_dot0, t,
            mass * q_dot0**2,  # Initial kinetic energy component
            spring_k * q0**2,  # Initial potential energy component
            np.sqrt(spring_k / mass),  # Angular frequency
            np.sqrt(mass),  # Mass factor
            np.sqrt(spring_k)  # Spring factor
        ], dtype=torch.float32)
        
        # Analytical solution
        omega = np.sqrt(spring_k / mass)
        position = q0 * np.cos(omega * t) + (q_dot0 / omega) * np.sin(omega * t)
        velocity = -q0 * omega * np.sin(omega * t) + q_dot0 * np.cos(omega * t)
        acceleration = -omega**2 * position
        
        # Energies
        kinetic_energy = 0.5 * mass * velocity**2
        potential_energy = 0.5 * spring_k * position**2
        total_energy = kinetic_energy + potential_energy
        lagrangian = kinetic_energy - potential_energy
        
        labels = {
            'position': torch.tensor(position, dtype=torch.float32),
            'velocity': torch.tensor(velocity, dtype=torch.float32),
            'acceleration': torch.tensor(acceleration, dtype=torch.float32),
            'kinetic_energy': torch.tensor(kinetic_energy, dtype=torch.float32),
            'potential_energy': torch.tensor(potential_energy, dtype=torch.float32),
            'total_energy': torch.tensor(total_energy, dtype=torch.float32),
            'lagrangian': torch.tensor(lagrangian, dtype=torch.float32)
        }
        
        return features, labels
    
    def _generate_pendulum(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate pendulum problem (small angle approximation)"""
        # System parameters
        length = random.uniform(0.5, 2.0)
        mass = random.uniform(0.5, 3.0)
        gravity = 9.81
        
        # Initial conditions (small angles)
        theta0 = random.uniform(-0.3, 0.3)  # Small angle approximation
        theta_dot0 = random.uniform(-1.0, 1.0)
        
        # Time
        t = random.uniform(0, 4 * np.pi)
        
        # Features
        omega = np.sqrt(gravity / length)
        features = torch.tensor([
            length, mass, gravity, theta0, theta_dot0, t,
            omega,  # Natural frequency
            mass * length**2,  # Moment of inertia
            mass * gravity * length,  # Restoring force factor
            length * theta0  # Arc length displacement
        ], dtype=torch.float32)
        
        # Small angle solution
        theta = theta0 * np.cos(omega * t) + (theta_dot0 / omega) * np.sin(omega * t)
        theta_dot = -theta0 * omega * np.sin(omega * t) + theta_dot0 * np.cos(omega * t)
        theta_ddot = -omega**2 * theta
        
        # Energies
        kinetic_energy = 0.5 * mass * length**2 * theta_dot**2
        potential_energy = 0.5 * mass * gravity * length * theta**2  # Small angle approx
        total_energy = kinetic_energy + potential_energy
        lagrangian = kinetic_energy - potential_energy
        
        labels = {
            'position': torch.tensor(theta, dtype=torch.float32),
            'velocity': torch.tensor(theta_dot, dtype=torch.float32),
            'acceleration': torch.tensor(theta_ddot, dtype=torch.float32),
            'kinetic_energy': torch.tensor(kinetic_energy, dtype=torch.float32),
            'potential_energy': torch.tensor(potential_energy, dtype=torch.float32),
            'total_energy': torch.tensor(total_energy, dtype=torch.float32),
            'lagrangian': torch.tensor(lagrangian, dtype=torch.float32)
        }
        
        return features, labels
    
    def _generate_particle_in_potential(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate particle in quadratic potential"""
        # System parameters
        mass = random.uniform(0.5, 3.0)
        potential_strength = random.uniform(1.0, 10.0)
        
        # Initial conditions
        x0 = random.uniform(-2.0, 2.0)
        v0 = random.uniform(-2.0, 2.0)
        
        # Time
        t = random.uniform(0, 2 * np.pi)
        
        # Features
        omega = np.sqrt(2 * potential_strength / mass)
        features = torch.tensor([
            mass, potential_strength, x0, v0, t,
            omega,  # Characteristic frequency
            mass * v0**2,  # Initial kinetic energy
            potential_strength * x0**2,  # Initial potential energy
            np.sqrt(mass * potential_strength),  # Coupling strength
            x0 / np.sqrt(mass)  # Scaled position
        ], dtype=torch.float32)
        
        # Solution for V(x) = (1/2) * potential_strength * x^2
        x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
        v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
        a = -omega**2 * x
        
        # Energies
        kinetic_energy = 0.5 * mass * v**2
        potential_energy = 0.5 * potential_strength * x**2
        total_energy = kinetic_energy + potential_energy
        lagrangian = kinetic_energy - potential_energy
        
        labels = {
            'position': torch.tensor(x, dtype=torch.float32),
            'velocity': torch.tensor(v, dtype=torch.float32),
            'acceleration': torch.tensor(a, dtype=torch.float32),
            'kinetic_energy': torch.tensor(kinetic_energy, dtype=torch.float32),
            'potential_energy': torch.tensor(potential_energy, dtype=torch.float32),
            'total_energy': torch.tensor(total_energy, dtype=torch.float32),
            'lagrangian': torch.tensor(lagrangian, dtype=torch.float32)
        }
        
        return features, labels
    
    def _compute_mechanics_difficulty(self, system_type: str, features: torch.Tensor) -> float:
        """Compute difficulty for mechanics problems"""
        difficulty = 0.0
        
        # System complexity
        if system_type == 'harmonic_oscillator':
            difficulty += 0.2
        elif system_type == 'pendulum':
            difficulty += 0.4
        elif system_type == 'particle_in_potential':
            difficulty += 0.3
        
        # Parameter-based difficulty
        if len(features) > 5:
            # Large time values are harder
            t = float(features[4]) if len(features) > 4 else 0
            difficulty += min(0.2, t / (4 * np.pi))
            
            # High frequency systems are harder
            if len(features) > 6:
                omega = float(features[6]) if features[6] > 0 else 1
                difficulty += min(0.2, omega / 10.0)
        
        return max(0.0, min(1.0, difficulty))

class CurriculumSampler(Sampler):
    """Curriculum learning sampler that orders samples by difficulty"""
    
    def __init__(self, dataset: BaseEulerDataset, curriculum_stage: str = 'basic'):
        self.dataset = dataset
        self.curriculum_stage = curriculum_stage
        self.difficulty_thresholds = {
            'basic': (0.0, 0.3),
            'intermediate': (0.2, 0.7),
            'advanced': (0.5, 1.0),
            'mixed': (0.0, 1.0)
        }
        
        self._create_curriculum_indices()
    
    def _create_curriculum_indices(self):
        """Create indices based on difficulty thresholds"""
        min_diff, max_diff = self.difficulty_thresholds[self.curriculum_stage]
        
        self.indices = []
        for i, sample in enumerate(self.dataset.data_samples):
            if min_diff <= sample.difficulty_level <= max_diff:
                self.indices.append(i)
        
        # Sort by difficulty (ascending for curriculum learning)
        self.indices.sort(key=lambda i: self.dataset.data_samples[i].difficulty_level)
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)

class MultiDomainDataset(Dataset):
    """Combined dataset for multi-domain training"""
    
    def __init__(self, config: EulerNetConfig, split: str = 'train'):
        self.config = config
        self.split = split
        
        # Create individual domain datasets
        self.number_theory_dataset = NumberTheoryDataset(config, split)
        self.analysis_dataset = AnalysisDataset(config, split)
        self.mechanics_dataset = MechanicsDataset(config, split)
        
        # Create unified sample list with domain labels
        self.unified_samples = []
        
        # Add number theory samples
        for sample in self.number_theory_dataset.data_samples:
            unified_sample = {
                'domain': 'number_theory',
                'sample': sample,
                'domain_id': 0
            }
            self.unified_samples.append(unified_sample)
        
        # Add analysis samples
        for sample in self.analysis_dataset.data_samples:
            unified_sample = {
                'domain': 'analysis',
                'sample': sample,
                'domain_id': 1
            }
            self.unified_samples.append(unified_sample)
        
        # Add mechanics samples
        for sample in self.mechanics_dataset.data_samples:
            unified_sample = {
                'domain': 'mechanics',
                'sample': sample,
                'domain_id': 2
            }
            self.unified_samples.append(unified_sample)
        
        # Shuffle for mixed training
        if split == 'train':
            random.shuffle(self.unified_samples)
    
    def __len__(self):
        return len(self.unified_samples)
    
    def __getitem__(self, idx):
        return self.unified_samples[idx]
    
    def get_domain_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across domains"""
        distribution = {'number_theory': 0, 'analysis': 0, 'mechanics': 0}
        
        for unified_sample in self.unified_samples:
            distribution[unified_sample['domain']] += 1
        
        return distribution

class DatasetManager:
    """Professional dataset management with caching and validation"""
    
    def __init__(self, config: EulerNetConfig):
        self.config = config
        self.cache_dir = Path('data_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # Dataset registry
        self.datasets = {}
        self.data_loaders = {}
        
    def create_datasets(self, force_regenerate: bool = False) -> Dict[str, Dataset]:
        """Create or load cached datasets"""
        cache_file = self.cache_dir / f"datasets_{self.config.get_model_signature()}.pkl"
        
        if not force_regenerate and cache_file.exists():
            logger.info(f"Loading cached datasets from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.datasets = pickle.load(f)
        else:
            logger.info("Creating new datasets...")
            
            # Create datasets for each split
            for split in ['train', 'val', 'test']:
                self.datasets[split] = MultiDomainDataset(self.config, split)
            
            # Cache the datasets
            with open(cache_file, 'wb') as f:
                pickle.dump(self.datasets, f)
            
            logger.info(f"Datasets cached to {cache_file}")
        
        # Log dataset statistics
        self._log_dataset_stats()
        
        return self.datasets
    
    def create_data_loaders(self, curriculum_stage: str = 'mixed') -> Dict[str, DataLoader]:
        """Create data loaders with optional curriculum learning"""
        if not self.datasets:
            self.create_datasets()
        
        self.data_loaders = {}
        
        for split in ['train', 'val', 'test']:
            dataset = self.datasets[split]
            
            # Use curriculum sampler for training
            sampler = None
            shuffle = True
            
            if split == 'train' and self.config.training.curriculum_learning and curriculum_stage != 'mixed':
                sampler = CurriculumSampler(dataset, curriculum_stage)
                shuffle = False  # Don't shuffle when using custom sampler
            
            self.data_loaders[split] = DataLoader(
                dataset=dataset,
                batch_size=self.config.training.batch_size if split == 'train' 
                          else self.config.training.batch_size // 2,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.data.pin_memory,
                drop_last=True if split == 'train' else False
            )
        
        return self.data_loaders
    
    def _log_dataset_stats(self):
        """Log comprehensive dataset statistics"""
        logger.info("Dataset Statistics:")
        logger.info("=" * 50)
        
        for split, dataset in self.datasets.items():
            domain_dist = dataset.get_domain_distribution()
            total_samples = len(dataset)
            
            logger.info(f"{split.upper()} SET:")
            logger.info(f"  Total samples: {total_samples}")
            logger.info(f"  Domain distribution:")
            
            for domain, count in domain_dist.items():
                percentage = (count / total_samples) * 100
                logger.info(f"    {domain}: {count} ({percentage:.1f}%)")
            
            # Difficulty distribution
            difficulties = [sample['sample'].difficulty_level for sample in dataset.unified_samples]
            if difficulties:
                logger.info(f"  Difficulty range: {min(difficulties):.3f} - {max(difficulties):.3f}")
                logger.info(f"  Average difficulty: {np.mean(difficulties):.3f}")
            
            logger.info("")
    
    def validate_datasets(self) -> Dict[str, bool]:
        """Validate dataset integrity and mathematical correctness"""
        validation_results = {}
        
        for split, dataset in self.datasets.items():
            logger.info(f"Validating {split} dataset...")
            
            validation_results[split] = True
            sample_count = min(100, len(dataset))  # Validate subset for efficiency
            
            for i in range(sample_count):
                try:
                    unified_sample = dataset[i]
                    sample = unified_sample['sample']
                    domain = unified_sample['domain']
                    
                    # Check tensor shapes and types
                    assert isinstance(sample.features, torch.Tensor)
                    assert isinstance(sample.labels, dict)
                    assert sample.features.dtype == torch.float32
                    
                    # Domain-specific validations
                    if domain == 'number_theory':
                        self._validate_number_theory_sample(sample)
                    elif domain == 'analysis':
                        self._validate_analysis_sample(sample)
                    elif domain == 'mechanics':
                        self._validate_mechanics_sample(sample)
                    
                except Exception as e:
                    logger.error(f"Validation failed for {split} sample {i}: {e}")
                    validation_results[split] = False
                    break
            
            if validation_results[split]:
                logger.info(f"✅ {split} dataset validation passed")
            else:
                logger.error(f"❌ {split} dataset validation failed")
        
        return validation_results
    
    def _validate_number_theory_sample(self, sample: DataSample):
        """Validate number theory sample"""
        from .utils import MATH_CONSTANTS
        
        # Check feature consistency
        n = int(sample.metadata['original_number'])
        
        # Validate prime classification
        computed_is_prime = MATH_CONSTANTS.is_prime(n)
        predicted_is_prime = bool(sample.labels['is_prime'].item() > 0.5)
        assert computed_is_prime == predicted_is_prime, f"Prime classification mismatch for {n}"
        
        # Validate totient function
        computed_totient = MATH_CONSTANTS.euler_totient(n)
        predicted_totient = int(sample.labels['totient'].item())
        assert abs(computed_totient - predicted_totient) < 1, f"Totient mismatch for {n}"
    
    def _validate_analysis_sample(self, sample: DataSample):
        """Validate analysis sample"""
        s = sample.metadata['s_value']
        n = sample.metadata['n_value']
        
        # Check zeta function value reasonableness
        zeta_value = sample.labels['zeta_value'].item()
        assert zeta_value > 1.0, f"Zeta value too small for s={s}"
        
        # Check harmonic number monotonicity
        harmonic = sample.labels['harmonic_number'].item()
        assert harmonic > 0, f"Harmonic number should be positive for n={n}"
    
    def _validate_mechanics_sample(self, sample: DataSample):
        """Validate mechanics sample"""
        # Check energy conservation (total energy should be positive)
        if 'total_energy' in sample.labels:
            total_energy = sample.labels['total_energy'].item()
            assert total_energy >= 0, "Total energy should be non-negative"
        
        # Check kinetic and potential energy non-negativity
        if 'kinetic_energy' in sample.labels:
            ke = sample.labels['kinetic_energy'].item()
            assert ke >= 0, "Kinetic energy should be non-negative"
        
        if 'potential_energy' in sample.labels:
            pe = sample.labels['potential_energy'].item()
            assert pe >= 0, "Potential energy should be non-negative"

def create_data_loaders(config: EulerNetConfig, curriculum_stage: str = 'mixed') -> Dict[str, DataLoader]:
    """Factory function to create data loaders"""
    manager = DatasetManager(config)
    return manager.create_data_loaders(curriculum_stage)

# Export functions and classes
__all__ = [
    'BaseEulerDataset',
    'NumberTheoryDataset', 
    'AnalysisDataset',
    'MechanicsDataset',
    'MultiDomainDataset',
    'CurriculumSampler',
    'DatasetManager',
    'DataSample',
    'create_data_loaders'
]

if __name__ == "__main__":
    # Test the dataset creation
    from config import EulerNetConfig
    
    config = EulerNetConfig()
    config.data.train_size = 1000
    config.data.val_size = 200
    config.data.test_size = 100
    
    manager = DatasetManager(config)
    
    # Create and validate datasets
    datasets = manager.create_datasets()
    validation_results = manager.validate_datasets()
    
    # Create data loaders
    data_loaders = manager.create_data_loaders()
    
    print("Dataset creation and validation completed!")
    print("Validation results:", validation_results)
    
    # Test a batch
    train_loader = data_loaders['train']
    for batch in train_loader:
        print(f"Batch domain: {batch[0]['domain']}")
        print(f"Feature shape: {batch[0]['sample'].features.shape}")
        break