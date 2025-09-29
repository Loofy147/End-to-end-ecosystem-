# complete_training_notebook.py
# Complete Working EulerNet Training Example
# This is a self-contained notebook that demonstrates the full EulerNet pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from tqdm import tqdm
import math
import random
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("EulerNet: Complete Training Pipeline")
print("=" * 50)

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

class SimpleConfig:
    """Simplified configuration for demonstration"""
    def __init__(self):
        # Model parameters
        self.d_model = 256
        self.n_heads = 8
        self.dropout_rate = 0.1
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.num_epochs = 20
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Data parameters
        self.train_size = 1000
        self.val_size = 200
        self.test_size = 100
        self.max_number = 1000
        
        # Mathematical parameters
        self.number_theory_input_size = 30
        self.analysis_input_size = 8
        self.mechanics_input_size = 6

config = SimpleConfig()
print(f"Using device: {config.device}")

# ============================================================================
# SECTION 2: MATHEMATICAL UTILITIES
# ============================================================================

class MathUtils:
    """Mathematical utility functions"""
    
    @staticmethod
    def is_prime(n):
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
    def euler_totient(n):
        """Euler's totient function"""
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
    
    @staticmethod
    def harmonic_number(n):
        """Compute harmonic number"""
        return sum(1.0 / k for k in range(1, n + 1))
    
    @staticmethod
    def simple_zeta(s, max_terms=1000):
        """Simple zeta function approximation"""
        if s <= 1:
            return float('inf')
        return sum(1.0 / (k ** s) for k in range(1, max_terms + 1))

math_utils = MathUtils()

# ============================================================================
# SECTION 3: DATASET IMPLEMENTATION
# ============================================================================

class EulerDataset(Dataset):
    """Complete dataset for all mathematical domains"""
    
    def __init__(self, size, max_number, split='train'):
        self.size = size
        self.max_number = max_number
        self.split = split
        self.samples = self._generate_samples()
    
    def _generate_samples(self):
        samples = []
        
        for i in range(self.size):
            # Number theory sample
            n = random.randint(2, self.max_number)
            nt_features, nt_labels = self._create_number_theory_sample(n)
            
            # Analysis sample
            s = random.uniform(1.1, 5.0)
            harmonic_n = random.randint(1, 100)
            analysis_features, analysis_labels = self._create_analysis_sample(s, harmonic_n)
            
            # Mechanics sample (harmonic oscillator)
            mech_features, mech_labels = self._create_mechanics_sample()
            
            samples.append({
                'number_theory': (nt_features, nt_labels),
                'analysis': (analysis_features, analysis_labels),
                'mechanics': (mech_features, mech_labels)
            })
        
        return samples
    
    def _create_number_theory_sample(self, n):
        """Create number theory sample"""
        features = []
        
        # Basic features
        features.append(float(n))
        features.append(float(math.log(n)))
        features.append(float(math.sqrt(n)))
        
        # Binary representation (16 bits)
        binary = format(n, '016b')
        features.extend([float(bit) for bit in binary])
        
        # Divisibility features
        for p in [2, 3, 5, 7]:
            features.append(float(n % p == 0))
            features.append(float(n % p))
        
        # Pad to fixed size
        while len(features) < config.number_theory_input_size:
            features.append(0.0)
        
        features = torch.tensor(features[:config.number_theory_input_size], dtype=torch.float32)
        
        # Labels
        labels = {
            'is_prime': torch.tensor(float(math_utils.is_prime(n))),
            'totient': torch.tensor(float(math_utils.euler_totient(n))),
            'log_totient': torch.tensor(float(math.log(math_utils.euler_totient(n) + 1)))
        }
        
        return features, labels
    
    def _create_analysis_sample(self, s, n):
        """Create analysis sample"""
        features = [
            s, s - 1, 1.0 / s, math.log(s),
            float(n), math.log(n), 1.0 / n, math.sqrt(n)
        ]
        
        features = torch.tensor(features, dtype=torch.float32)
        
        # Labels
        zeta_val = math_utils.simple_zeta(s, 500)
        harmonic_val = math_utils.harmonic_number(n)
        gamma_approx = harmonic_val - math.log(n)
        
        labels = {
            'zeta': torch.tensor(float(zeta_val)),
            'harmonic': torch.tensor(float(harmonic_val)),
            'gamma_approx': torch.tensor(float(gamma_approx))
        }
        
        return features, labels
    
    def _create_mechanics_sample(self):
        """Create mechanics sample (harmonic oscillator)"""
        # System parameters
        mass = random.uniform(0.5, 2.0)
        spring_k = random.uniform(1.0, 10.0)
        x0 = random.uniform(-2.0, 2.0)
        v0 = random.uniform(-2.0, 2.0)
        t = random.uniform(0, 2 * math.pi)
        
        omega = math.sqrt(spring_k / mass)
        
        features = torch.tensor([mass, spring_k, x0, v0, t, omega], dtype=torch.float32)
        
        # Analytical solution
        x = x0 * math.cos(omega * t) + (v0 / omega) * math.sin(omega * t)
        v = -x0 * omega * math.sin(omega * t) + v0 * math.cos(omega * t)
        
        ke = 0.5 * mass * v**2
        pe = 0.5 * spring_k * x**2
        
        labels = {
            'position': torch.tensor(float(x)),
            'velocity': torch.tensor(float(v)),
            'kinetic_energy': torch.tensor(float(ke)),
            'potential_energy': torch.tensor(float(pe))
        }
        
        return features, labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# ============================================================================
# SECTION 4: MODEL ARCHITECTURE
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention for mathematical relationships"""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.W_o(attn_output)

class NumberTheoryNetwork(nn.Module):
    """Network for number theory problems"""
    
    def __init__(self, input_size, d_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        
        self.prime_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.totient_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive
        )
        
        self.log_totient_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        return {
            'is_prime': self.prime_head(encoded).squeeze(-1),
            'totient': self.totient_head(encoded).squeeze(-1),
            'log_totient': self.log_totient_head(encoded).squeeze(-1)
        }

class AnalysisNetwork(nn.Module):
    """Network for mathematical analysis"""
    
    def __init__(self, input_size, d_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        
        self.zeta_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        self.harmonic_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        self.gamma_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        return {
            'zeta': self.zeta_head(encoded).squeeze(-1),
            'harmonic': self.harmonic_head(encoded).squeeze(-1),
            'gamma_approx': self.gamma_head(encoded).squeeze(-1)
        }

class MechanicsNetwork(nn.Module):
    """Network for mechanics problems"""
    
    def __init__(self, input_size, d_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.Tanh(),  # Smooth for physics
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        
        self.position_head = nn.Linear(d_model, 1)
        self.velocity_head = nn.Linear(d_model, 1)
        
        self.kinetic_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        self.potential_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        return {
            'position': self.position_head(encoded).squeeze(-1),
            'velocity': self.velocity_head(encoded).squeeze(-1),
            'kinetic_energy': self.kinetic_head(encoded).squeeze(-1),
            'potential_energy': self.potential_head(encoded).squeeze(-1)
        }

class EulerNet(nn.Module):
    """Complete EulerNet architecture"""
    
    def __init__(self, config):
        super().__init__()
        
        # Domain-specific networks
        self.number_theory_net = NumberTheoryNetwork(
            config.number_theory_input_size, config.d_model
        )
        self.analysis_net = AnalysisNetwork(
            config.analysis_input_size, config.d_model
        )
        self.mechanics_net = MechanicsNetwork(
            config.mechanics_input_size, config.d_model
        )
        
        # Cross-domain attention
        self.cross_attention = MultiHeadAttention(config.d_model, config.n_heads)
        
    def forward(self, batch):
        results = {}
        
        # Process each domain
        if 'number_theory' in batch:
            nt_features, _ = batch['number_theory']
            results['number_theory'] = self.number_theory_net(nt_features)
        
        if 'analysis' in batch:
            analysis_features, _ = batch['analysis']
            results['analysis'] = self.analysis_net(analysis_features)
        
        if 'mechanics' in batch:
            mech_features, _ = batch['mechanics']
            results['mechanics'] = self.mechanics_net(mech_features)
        
        return results

# ============================================================================
# SECTION 5: TRAINING PIPELINE
# ============================================================================

class EulerTrainer:
    """Complete training pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create model
        self.model = EulerNet(config).to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def compute_loss(self, predictions, labels):
        """Compute comprehensive loss"""
        total_loss = 0.0
        loss_components = {}
        
        # Number theory losses
        if 'number_theory' in predictions:
            nt_pred = predictions['number_theory']
            nt_labels = labels['number_theory'][1]  # Extract labels from tuple
            
            # Prime classification loss
            prime_loss = self.bce_loss(nt_pred['is_prime'], nt_labels['is_prime'])
            loss_components['prime_loss'] = prime_loss
            total_loss += prime_loss
            
            # Totient loss (use log for stability)
            log_totient_loss = self.mse_loss(nt_pred['log_totient'], nt_labels['log_totient'])
            loss_components['log_totient_loss'] = log_totient_loss
            total_loss += log_totient_loss
        
        # Analysis losses
        if 'analysis' in predictions:
            analysis_pred = predictions['analysis']
            analysis_labels = labels['analysis'][1]
            
            # Zeta function loss
            zeta_loss = self.mse_loss(analysis_pred['zeta'], analysis_labels['zeta'])
            loss_components['zeta_loss'] = zeta_loss
            total_loss += zeta_loss
            
            # Harmonic number loss
            harmonic_loss = self.mse_loss(analysis_pred['harmonic'], analysis_labels['harmonic'])
            loss_components['harmonic_loss'] = harmonic_loss
            total_loss += harmonic_loss
            
            # Gamma approximation loss
            gamma_loss = self.mse_loss(analysis_pred['gamma_approx'], analysis_labels['gamma_approx'])
            loss_components['gamma_loss'] = gamma_loss
            total_loss += gamma_loss
        
        # Mechanics losses
        if 'mechanics' in predictions:
            mech_pred = predictions['mechanics']
            mech_labels = labels['mechanics'][1]
            
            # Position and velocity losses
            pos_loss = self.mse_loss(mech_pred['position'], mech_labels['position'])
            vel_loss = self.mse_loss(mech_pred['velocity'], mech_labels['velocity'])
            
            loss_components['position_loss'] = pos_loss
            loss_components['velocity_loss'] = vel_loss
            total_loss += pos_loss + vel_loss
            
            # Energy losses
            ke_loss = self.mse_loss(mech_pred['kinetic_energy'], mech_labels['kinetic_energy'])
            pe_loss = self.mse_loss(mech_pred['potential_energy'], mech_labels['potential_energy'])
            
            loss_components['kinetic_loss'] = ke_loss
            loss_components['potential_loss'] = pe_loss
            total_loss += ke_loss + pe_loss
            
            # Energy conservation loss
            total_energy_pred = mech_pred['kinetic_energy'] + mech_pred['potential_energy']
            total_energy_true = mech_labels['kinetic_energy'] + mech_labels['potential_energy']
            conservation_loss = self.mse_loss(total_energy_pred, total_energy_true)
            
            loss_components['conservation_loss'] = conservation_loss
            total_loss += 0.5 * conservation_loss  # Weight conservation loss
        
        loss_components['total_loss'] = total_loss
        return total_loss, loss_components
    
    def compute_accuracy(self, predictions, labels):
        """Compute accuracy metrics"""
        metrics = {}
        
        if 'number_theory' in predictions:
            nt_pred = predictions['number_theory']
            nt_labels = labels['number_theory'][1]
            
            # Prime classification accuracy
            prime_pred_binary = (nt_pred['is_prime'] > 0.5).float()
            prime_acc = (prime_pred_binary == nt_labels['is_prime']).float().mean()
            metrics['prime_accuracy'] = prime_acc.item()
        
        return metrics
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_metrics = {}
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            device_batch = {}
            for domain, (features, labels) in batch.items():
                device_batch[domain] = (
                    features.to(self.device),
                    {k: v.to(self.device) for k, v in labels.items()}
                )
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(device_batch)
            
            # Compute loss
            loss, loss_components = self.compute_loss(predictions, device_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            batch_metrics = self.compute_accuracy(predictions, device_batch)
            
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = []
                total_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in total_metrics.items()}
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, avg_metrics
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move batch to device
                device_batch = {}
                for domain, (features, labels) in batch.items():
                    device_batch[domain] = (
                        features.to(self.device),
                        {k: v.to(self.device) for k, v in labels.items()}
                    )
                
                # Forward pass
                predictions = self.model(device_batch)
                
                # Compute loss and metrics
                loss, _ = self.compute_loss(predictions, device_batch)
                batch_metrics = self.compute_accuracy(predictions, device_batch)
                
                total_loss += loss.item()
                for key, value in batch_metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = []
                    total_metrics[key].append(value)
        
        avg_metrics = {k: np.mean(v) for k, v in total_metrics.items()}
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, avg_metrics
    
    def train(self, train_loader, val_loader):
        """Complete training loop"""
        print("Starting EulerNet Training...")
        print("=" * 50)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 30)
            
            # Training
            start_time = time.time()
            train_loss, train_metrics = self.train_epoch(train_loader)
            train_time = time.time() - start_time
            
            # Validation
            start_time = time.time()
            val_loss, val_metrics = self.validate_epoch(val_loader)
            val_time = time.time() - start_time
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if 'prime_accuracy' in train_metrics:
                self.history['train_acc'].append(train_metrics['prime_accuracy'])
            if 'prime_accuracy' in val_metrics:
                self.history['val_acc'].append(val_metrics['prime_accuracy'])
            
            # Print epoch results
            print(f"Training   - Loss: {train_loss:.4f}, Time: {train_time:.2f}s")
            print(f"Validation - Loss: {val_loss:.4f}, Time: {val_time:.2f}s")
            
            if train_metrics:
                print(f"Train Metrics: {train_metrics}")
            if val_metrics:
                print(f"Val Metrics: {val_metrics}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
        
        print("\nTraining completed!")
        return self.history
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        Path('checkpoints').mkdir(exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, f'checkpoints/{filename}')
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.history['train_loss'], label='Training Loss', color='blue')
        ax1.plot(self.history['val_loss'], label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves (if available)
        if self.history['train_acc'] and self.history['val_acc']:
            ax2.plot(self.history['train_acc'], label='Training Accuracy', color='blue')
            ax2.plot(self.history['val_acc'], label='Validation Accuracy', color='red')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Prime Classification Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Accuracy data not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Prime Classification Accuracy')
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()

# ============================================================================
# SECTION 6: MATHEMATICAL VALIDATION
# ============================================================================

def validate_mathematical_correctness(model, test_loader, device):
    """Validate mathematical correctness of predictions"""
    print("\nValidating Mathematical Correctness...")
    print("=" * 40)
    
    model.eval()
    correct_classifications = 0
    total_samples = 0
    
    totient_errors = []
    zeta_errors = []
    energy_conservation_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            device_batch = {}
            for domain, (features, labels) in batch.items():
                device_batch[domain] = (
                    features.to(device),
                    {k: v.to(device) for k, v in labels.items()}
                )
            
            predictions = model(device_batch)
            
            # Validate number theory predictions
            if 'number_theory' in predictions:
                nt_pred = predictions['number_theory']
                nt_labels = device_batch['number_theory'][1]
                
                # Prime classification
                prime_pred = (nt_pred['is_prime'] > 0.5).cpu().numpy()
                prime_true = nt_labels['is_prime'].cpu().numpy()
                
                correct_classifications += np.sum(prime_pred == prime_true)
                total_samples += len(prime_pred)
                
                # Totient function relative errors
                totient_pred = nt_pred['totient'].cpu().numpy()
                totient_true = nt_labels['totient'].cpu().numpy()
                relative_errors = np.abs(totient_pred - totient_true) / (totient_true + 1e-8)
                totient_errors.extend(relative_errors)
            
            # Validate analysis predictions
            if 'analysis' in predictions:
                analysis_pred = predictions['analysis']
                analysis_labels = device_batch['analysis'][1]
                
                zeta_pred = analysis_pred['zeta'].cpu().numpy()
                zeta_true = analysis_labels['zeta'].cpu().numpy()
                zeta_rel_errors = np.abs(zeta_pred - zeta_true) / (zeta_true + 1e-8)
                zeta_errors.extend(zeta_rel_errors)
            
            # Validate mechanics predictions (energy conservation)
            if 'mechanics' in predictions:
                mech_pred = predictions['mechanics']
                ke_pred = mech_pred['kinetic_energy'].cpu().numpy()
                pe_pred = mech_pred['potential_energy'].cpu().numpy()
                
                # Check energy conservation (should be approximately constant)
                total_energy = ke_pred + pe_pred
                energy_variance = np.var(total_energy)
                energy_conservation_errors.extend([energy_variance])
    
    # Print validation results
    if total_samples > 0:
        prime_accuracy = correct_classifications / total_samples
        print(f"Prime Classification Accuracy: {prime_accuracy:.3f}")
    
    if totient_errors:
        avg_totient_error = np.mean(totient_errors)
        print(f"Average Totient Relative Error: {avg_totient_error:.4f}")
    
    if zeta_errors:
        avg_zeta_error = np.mean(zeta_errors)
        print(f"Average Zeta Function Relative Error: {avg_zeta_error:.4f}")
    
    if energy_conservation_errors:
        avg_energy_error = np.mean(energy_conservation_errors)
        print(f"Average Energy Conservation Error: {avg_energy_error:.6f}")

# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("Initializing EulerNet Training Pipeline...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = EulerDataset(config.train_size, config.max_number, 'train')
    val_dataset = EulerDataset(config.val_size, config.max_number, 'val')
    test_dataset = EulerDataset(config.test_size, config.max_number, 'test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create trainer
    trainer = EulerTrainer(config)
    
    # Print model information
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Train the model
    history = trainer.train(train_loader, val_loader)
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Mathematical validation
    validate_mathematical_correctness(trainer.model, test_loader, trainer.device)
    
    # Test some specific examples
    print("\nTesting Specific Mathematical Examples...")
    print("=" * 40)
    
    # Test prime numbers
    test_numbers = [17, 25, 31, 49, 67, 100]
    
    trainer.model.eval()
    with torch.no_grad():
        for n in test_numbers:
            # Create sample
            features = []
            features.append(float(n))
            features.append(float(math.log(n)))
            features.append(float(math.sqrt(n)))
            
            binary = format(n, '016b')
            features.extend([float(bit) for bit in binary])
            
            for p in [2, 3, 5, 7]:
                features.append(float(n % p == 0))
                features.append(float(n % p))
            
            while len(features) < config.number_theory_input_size:
                features.append(0.0)
            
            features_tensor = torch.tensor(features[:config.number_theory_input_size], dtype=torch.float32).unsqueeze(0).to(trainer.device)
            
            # Make prediction
            batch = {'number_theory': (features_tensor, {})}
            pred = trainer.model(batch)['number_theory']
            
            is_prime_pred = pred['is_prime'].item() > 0.5
            totient_pred = pred['totient'].item()
            
            # Ground truth
            is_prime_true = math_utils.is_prime(n)
            totient_true = math_utils.euler_totient(n)
            
            print(f"n={n:3d}: Prime? Pred={is_prime_pred}, True={is_prime_true}, "
                  f"φ(n): Pred={totient_pred:.1f}, True={totient_true}")
    
    print("\n" + "=" * 50)
    print("EulerNet Training Pipeline Completed Successfully!")
    print("=" * 50)
    
    return trainer, history

# Run the complete training pipeline
if __name__ == "__main__":
    trainer, history = main()