# models.py
# EulerNet Neural Network Architectures
# Professional implementation of mathematical neural networks

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import logging

# Import the standalone config instead of the problematic one
try:
    from .standalone_config import EulerNetConfig
except ImportError:
    from standalone_config import EulerNetConfig

logger = logging.getLogger(__name__)

class MathematicallyInformedLayer(nn.Module):
    """Base class for mathematically-informed neural network layers"""
    
    def __init__(self, input_size: int, output_size: int, mathematical_constraint: Optional[str] = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mathematical_constraint = mathematical_constraint
        
        # Standard linear transformation
        self.linear = nn.Linear(input_size, output_size)
        
        # Mathematical constraint enforcement
        if mathematical_constraint:
            self._setup_constraint()
    
    def _setup_constraint(self):
        """Setup mathematical constraints"""
        if self.mathematical_constraint == 'positive':
            self.constraint_fn = F.softplus
        elif self.mathematical_constraint == 'probability':
            self.constraint_fn = torch.sigmoid
        elif self.mathematical_constraint == 'unit_norm':
            self.constraint_fn = F.normalize
        else:
            self.constraint_fn = lambda x: x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return self.constraint_fn(out)

class MultiHeadMathematicalAttention(nn.Module):
    """Multi-head attention specialized for mathematical relationships"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Mathematical relationship embeddings
        self.mathematical_bias = nn.Parameter(torch.zeros(n_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights using Xavier uniform for mathematical stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for mathematical stability"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations and reshape
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with mathematical bias
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_scores += self.mathematical_bias
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.W_o(attention_output)

class EulerTotientNetwork(nn.Module):
    """Specialized network for Euler's totient function φ(n)"""
    
    def __init__(self, config: EulerNetConfig):
        super().__init__()
        self.config = config
        input_size = config.model.number_theory_input_size
        
        # Feature extraction with mathematical structure
        self.feature_extractor = nn.Sequential(
            MathematicallyInformedLayer(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128) if config.model.use_batch_norm else nn.Identity(),
            nn.Dropout(config.model.dropout_rate),
            
            MathematicallyInformedLayer(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256) if config.model.use_batch_norm else nn.Identity(),
            nn.Dropout(config.model.dropout_rate),
            
            MathematicallyInformedLayer(256, 128),
            nn.ReLU(),
        )
        
        # Totient-specific processing
        # φ(n) has specific mathematical properties we can encode
        self.totient_processor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Output layer with positive constraint (φ(n) ≥ 1)
        self.totient_head = MathematicallyInformedLayer(32, 1, 'positive')
        
        # Mathematical consistency layers
        self.multiplicative_consistency = nn.Linear(128, 1)  # For φ(mn) = φ(m)φ(n) when gcd(m,n)=1
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract general features
        features = self.feature_extractor(x)
        
        # Process for totient function
        totient_features = self.totient_processor(features)
        totient_value = self.totient_head(totient_features)
        
        # Multiplicative property check
        multiplicative_score = torch.sigmoid(self.multiplicative_consistency(features))
        
        return {
            'totient': totient_value.squeeze(-1),
            'multiplicative_property': multiplicative_score.squeeze(-1),
            'features': features
        }

class EulerPrimeNetwork(nn.Module):
    """Network for prime classification and related number-theoretic properties"""
    
    def __init__(self, config: EulerNetConfig):
        super().__init__()
        self.config = config
        input_size = config.model.number_theory_input_size
        
        # Shared encoder with attention mechanism
        self.shared_encoder = nn.Sequential(
            MathematicallyInformedLayer(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256) if config.model.use_batch_norm else nn.Identity(),
            nn.Dropout(config.model.dropout_rate),
            
            MathematicallyInformedLayer(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512) if config.model.use_batch_norm else nn.Identity(),
            nn.Dropout(config.model.dropout_rate),
        )
        
        # Multi-head attention for number relationships
        if config.model.use_attention:
            self.attention = MultiHeadMathematicalAttention(512, config.model.n_heads)
        else:
            self.attention = nn.Identity()
        
        # Prime classification head
        self.prime_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.model.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Legendre symbol predictor (8 symbols for different primes)
        self.legendre_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(config.model.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Tanh()  # Legendre symbols are in {-1, 0, 1}
        )
        
        # Möbius function predictor
        self.mobius_predictor = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Möbius function is in {-1, 0, 1}
        )
        
        # Prime factorization complexity estimator
        self.factorization_complexity = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Always positive
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        
        # Shared encoding
        encoded = self.shared_encoder(x)
        
        # Apply attention if enabled
        if isinstance(self.attention, MultiHeadMathematicalAttention):
            # Reshape for attention (add sequence dimension)
            encoded = encoded.unsqueeze(1)  # [batch, 1, 512]
            attended = self.attention(encoded)
            attended = attended.squeeze(1)  # [batch, 512]
        else:
            attended = encoded
        
        # Multiple predictions
        predictions = {
            'is_prime': self.prime_classifier(attended).squeeze(-1),
            'legendre_symbols': self.legendre_predictor(attended),
            'mobius': self.mobius_predictor(attended).squeeze(-1),
            'factorization_complexity': self.factorization_complexity(attended).squeeze(-1),
            'features': attended
        }
        
        return predictions

class EulerZetaNetwork(nn.Module):
    """Network for Riemann zeta function and Euler products"""
    
    def __init__(self, config: EulerNetConfig):
        super().__init__()
        self.config = config
        input_size = config.model.analysis_input_size
        
        # Encoder with mathematical structure
        self.encoder = nn.Sequential(
            MathematicallyInformedLayer(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128) if config.model.use_batch_norm else nn.Identity(),
            nn.Dropout(config.model.dropout_rate),
            
            MathematicallyInformedLayer(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256) if config.model.use_batch_norm else nn.Identity(),
            nn.Dropout(config.model.dropout_rate),
            
            MathematicallyInformedLayer(256, 512),
            nn.ReLU(),
        )
        
        # Zeta function head (ζ(s) > 1 for s > 1)
        self.zeta_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(config.model.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            MathematicallyInformedLayer(64, 1, 'positive')
        )
        
        # Euler-Mascheroni constant approximation head
        self.gamma_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Can be positive or negative
        )
        
        # Harmonic number head (always positive)
        self.harmonic_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            MathematicallyInformedLayer(32, 1, 'positive')
        )
        
        # Euler product head
        self.euler_product_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(config.model.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            MathematicallyInformedLayer(64, 1, 'positive')
        )
        
        # Series convergence analyzer
        self.convergence_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Convergence probability
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode input
        encoded = self.encoder(x)
        
        # Multiple mathematical predictions
        predictions = {
            'zeta_value': self.zeta_head(encoded).squeeze(-1),
            'gamma_approximation': self.gamma_head(encoded).squeeze(-1),
            'harmonic_number': self.harmonic_head(encoded).squeeze(-1),
            'euler_product': self.euler_product_head(encoded).squeeze(-1),
            'convergence_probability': self.convergence_head(encoded).squeeze(-1),
            'features': encoded
        }
        
        return predictions

class EulerMechanicsNetwork(nn.Module):
    """Network for Lagrangian mechanics and generalized coordinates"""
    
    def __init__(self, config: EulerNetConfig):
        super().__init__()
        self.config = config
        input_size = config.model.mechanics_input_size
        
        # Physics-informed encoder
        self.physics_encoder = nn.Sequential(
            MathematicallyInformedLayer(input_size, 64),
            nn.Tanh(),  # Smooth activation for physics
            MathematicallyInformedLayer(64, 128),
            nn.Tanh(),
            MathematicallyInformedLayer(128, 256),
            nn.Tanh(),
        )
        
        # Position head (unbounded)
        self.position_head = nn.Linear(256, 1)
        
        # Velocity head (unbounded)
        self.velocity_head = nn.Linear(256, 1)
        
        # Acceleration head (unbounded)
        self.acceleration_head = nn.Linear(256, 1)
        
        # Lagrangian head (unbounded - can be positive or negative)
        self.lagrangian_head = nn.Linear(256, 1)
        
        # Energy heads (must be non-negative)
        self.kinetic_energy_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            MathematicallyInformedLayer(64, 1, 'positive')
        )
        
        self.potential_energy_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            MathematicallyInformedLayer(64, 1, 'positive')
        )
        
        # Total energy head (conservation check)
        self.total_energy_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            MathematicallyInformedLayer(32, 1, 'positive')
        )
        
        # Euler-Lagrange equation residual (should be zero)
        self.euler_lagrange_residual = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # Residual can be positive or negative
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Physics-informed encoding
        encoded = self.physics_encoder(x)
        
        # Mechanical quantities
        position = self.position_head(encoded)
        velocity = self.velocity_head(encoded)
        acceleration = self.acceleration_head(encoded)
        
        # Energies
        kinetic_energy = self.kinetic_energy_head(encoded)
        potential_energy = self.potential_energy_head(encoded)
        total_energy = self.total_energy_head(encoded)
        
        # Lagrangian and equation residual
        lagrangian = self.lagrangian_head(encoded)
        euler_lagrange_residual = self.euler_lagrange_residual(encoded)
        
        predictions = {
            'position': position.squeeze(-1),
            'velocity': velocity.squeeze(-1),
            'acceleration': acceleration.squeeze(-1),
            'kinetic_energy': kinetic_energy.squeeze(-1),
            'potential_energy': potential_energy.squeeze(-1),
            'total_energy': total_energy.squeeze(-1),
            'lagrangian': lagrangian.squeeze(-1),
            'euler_lagrange_residual': euler_lagrange_residual.squeeze(-1),
            'features': encoded
        }
        
        return predictions

class CrossDomainFusionNetwork(nn.Module):
    """Network for fusing information across mathematical domains"""
    
    def __init__(self, config: EulerNetConfig):
        super().__init__()
        self.config = config
        
        # Assume 512-dimensional features from each domain
        self.fusion_input_size = 512 * 3  # number_theory + analysis + mechanics
        
        # Cross-domain attention
        self.cross_attention = MultiHeadMathematicalAttention(512, config.model.n_heads)
        
        # Fusion layers
        self.fusion_network = nn.Sequential(
            nn.Linear(self.fusion_input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512) if config.model.use_batch_norm else nn.Identity(),
            nn.Dropout(config.model.dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256) if config.model.use_batch_norm else nn.Identity(),
            nn.Dropout(config.model.dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Mathematical consistency heads
        self.consistency_heads = nn.ModuleDict({
            'euler_identity': nn.Linear(128, 1),  # e^(iπ) + 1 = 0 verification
            'fundamental_theorem': nn.Linear(128, 1),  # Calculus fundamental theorem
            'conservation_laws': nn.Linear(128, 1),  # Energy conservation in mechanics
            'number_theory_analysis': nn.Linear(128, 1),  # Prime number theorem consistency
        })
        
    def forward(self, domain_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Fuse features from different mathematical domains
        
        Args:
            domain_features: Dict with keys 'number_theory', 'analysis', 'mechanics'
        """
        batch_size = next(iter(domain_features.values())).size(0)
        
        # Prepare features for cross-attention
        feature_list = []
        domain_names = []
        
        for domain, features in domain_features.items():
            if features is not None:
                feature_list.append(features)
                domain_names.append(domain)
        
        if len(feature_list) < 2:
            # Not enough domains for meaningful fusion
            return {'unified_features': torch.zeros(batch_size, 128)}
        
        # Stack features for attention
        stacked_features = torch.stack(feature_list, dim=1)  # [batch, n_domains, 512]
        
        # Apply cross-domain attention
        attended_features = self.cross_attention(stacked_features)  # [batch, n_domains, 512]
        
        # Flatten for fusion network
        flattened = attended_features.view(batch_size, -1)
        
        # Pad or truncate to expected size
        if flattened.size(1) < self.fusion_input_size:
            padding = torch.zeros(batch_size, self.fusion_input_size - flattened.size(1), device=flattened.device)
            flattened = torch.cat([flattened, padding], dim=1)
        elif flattened.size(1) > self.fusion_input_size:
            flattened = flattened[:, :self.fusion_input_size]
        
        # Fuse features
        unified_features = self.fusion_network(flattened)
        
        # Compute consistency scores
        consistency_scores = {}
        for consistency_type, head in self.consistency_heads.items():
            consistency_scores[consistency_type] = torch.sigmoid(head(unified_features)).squeeze(-1)
        
        return {
            'unified_features': unified_features,
            'consistency_scores': consistency_scores,
            'attended_features': attended_features
        }

class EulerNet(nn.Module):
    """Unified EulerNet architecture combining all mathematical domains"""
    
    def __init__(self, config: EulerNetConfig):
        super().__init__()
        self.config = config
        
        # Domain-specific networks
        self.totient_network = EulerTotientNetwork(config)
        self.prime_network = EulerPrimeNetwork(config)
        self.zeta_network = EulerZetaNetwork(config)
        self.mechanics_network = EulerMechanicsNetwork(config)
        
        # Cross-domain fusion
        self.fusion_network = CrossDomainFusionNetwork(config)
        
        # Mathematical consistency enforcer
        self.consistency_loss_weight = 0.1
        self.mathematical_consistency_loss = 0.0
        
        # Initialize weights
        self.apply(self._initialize_weights)
        
    def _initialize_weights(self, module):
        """Initialize weights using mathematically-informed strategies"""
        if isinstance(module, nn.Linear):
            if self.config.model.weight_initialization == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif self.config.model.weight_initialization == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif self.config.model.weight_initialization == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif self.config.model.weight_initialization == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Forward pass for different input types
        
        Args:
            inputs: Dictionary with domain keys and corresponding tensors
                   - 'number_theory': tensor for number theory problems
                   - 'analysis': tensor for analysis problems  
                   - 'mechanics': tensor for mechanics problems
        """
        results = {}
        domain_features = {}
        
        # Process each domain independently
        if 'number_theory' in inputs:
            nt_input = inputs['number_theory']
            
            # Totient function predictions
            totient_results = self.totient_network(nt_input)
            results['totient'] = totient_results
            
            # Prime and number theory predictions
            prime_results = self.prime_network(nt_input)
            results['prime_properties'] = prime_results
            
            # Store features for fusion
            domain_features['number_theory'] = prime_results['features']
        
        if 'analysis' in inputs:
            analysis_input = inputs['analysis']
            
            # Zeta function and analysis predictions
            zeta_results = self.zeta_network(analysis_input)
            results['zeta_analysis'] = zeta_results
            
            # Store features for fusion
            domain_features['analysis'] = zeta_results['features']
        
        if 'mechanics' in inputs:
            mechanics_input = inputs['mechanics']
            
            # Mechanics predictions
            mechanics_results = self.mechanics_network(mechanics_input)
            results['mechanics_solution'] = mechanics_results
            
            # Store features for fusion
            domain_features['mechanics'] = mechanics_results['features']
        
        # Cross-domain fusion if multiple domains present
        if len(domain_features) > 1:
            fusion_results = self.fusion_network(domain_features)
            results['fusion'] = fusion_results
            
            # Compute mathematical consistency loss
            self.mathematical_consistency_loss = self.compute_consistency_loss(results, fusion_results)
        
        return results
    
    def compute_consistency_loss(self, predictions: Dict[str, Any], fusion_results: Dict[str, Any]) -> torch.Tensor:
        """
        Enforce mathematical consistency across domains
        """
        consistency_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Euler's totient function consistency: φ(p) = p-1 for prime p
        if 'totient' in predictions and 'prime_properties' in predictions:
            totient_pred = predictions['totient']['totient']
            is_prime_pred = predictions['prime_properties']['is_prime']
            
            # For numbers classified as prime, φ(n) should be close to n-1
            # This is approximated since we don't have the original n directly
            prime_mask = (is_prime_pred > 0.5).float()
            # We can't directly compute n-1 without the original number,
            # but we can ensure totient values for primes are reasonable
            prime_totient_consistency = torch.mean(prime_mask * torch.relu(1.0 - totient_pred / 10.0))
            consistency_loss += prime_totient_consistency
        
        # Zeta function and Euler product consistency
        if 'zeta_analysis' in predictions:
            zeta_pred = predictions['zeta_analysis']['zeta_value']
            euler_product_pred = predictions['zeta_analysis']['euler_product']
            
            # Euler product should approximate zeta function
            zeta_euler_consistency = F.mse_loss(zeta_pred, euler_product_pred)
            consistency_loss += zeta_euler_consistency
        
        # Energy conservation in mechanics
        if 'mechanics_solution' in predictions:
            kinetic = predictions['mechanics_solution']['kinetic_energy']
            potential = predictions['mechanics_solution']['potential_energy']
            total_predicted = predictions['mechanics_solution']['total_energy']
            
            # Total energy should equal sum of kinetic and potential
            energy_conservation = F.mse_loss(total_predicted, kinetic + potential)
            consistency_loss += energy_conservation
            
            # Euler-Lagrange equation residual should be small
            euler_lagrange_residual = predictions['mechanics_solution']['euler_lagrange_residual']
            euler_lagrange_consistency = torch.mean(torch.square(euler_lagrange_residual))
            consistency_loss += euler_lagrange_consistency
        
        # Cross-domain consistency from fusion network
        if 'fusion' in fusion_results and 'consistency_scores' in fusion_results:
            consistency_scores = fusion_results['consistency_scores']
            
            # All consistency scores should be high (close to 1)
            for score in consistency_scores.values():
                consistency_loss += torch.mean(1.0 - score)
        
        return consistency_loss * self.consistency_loss_weight
    
    def get_mathematical_insights(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract mathematical insights from predictions
        """
        insights = {}
        
        # Number theory insights
        if 'prime_properties' in predictions:
            prime_pred = predictions['prime_properties']['is_prime']
            insights['estimated_prime_probability'] = torch.mean(prime_pred).item()
            
            if 'totient' in predictions:
                totient_pred = predictions['totient']['totient']
                insights['average_totient'] = torch.mean(totient_pred).item()
        
        # Analysis insights
        if 'zeta_analysis' in predictions:
            zeta_pred = predictions['zeta_analysis']['zeta_value']
            insights['average_zeta_value'] = torch.mean(zeta_pred).item()
            
            convergence_prob = predictions['zeta_analysis']['convergence_probability']
            insights['series_convergence_probability'] = torch.mean(convergence_prob).item()
        
        # Mechanics insights
        if 'mechanics_solution' in predictions:
            total_energy = predictions['mechanics_solution']['total_energy']
            insights['average_total_energy'] = torch.mean(total_energy).item()
            
            euler_lagrange_residual = predictions['mechanics_solution']['euler_lagrange_residual']
            insights['euler_lagrange_equation_error'] = torch.mean(torch.abs(euler_lagrange_residual)).item()
        
        # Consistency insights
        if hasattr(self, 'mathematical_consistency_loss'):
            insights['mathematical_consistency_loss'] = self.mathematical_consistency_loss.item()
        
        return insights
    
    def mathematical_validation(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        """
        Perform mathematical validation of model outputs
        """
        with torch.no_grad():
            predictions = self.forward(inputs)
            
        validation_results = {}
        
        # Validate positive constraints
        if 'totient' in predictions:
            totient_values = predictions['totient']['totient']
            validation_results['totient_positive'] = torch.all(totient_values >= 1.0).item()
        
        if 'zeta_analysis' in predictions:
            zeta_values = predictions['zeta_analysis']['zeta_value']
            validation_results['zeta_reasonable'] = torch.all(zeta_values >= 1.0).item()
            
            harmonic_values = predictions['zeta_analysis']['harmonic_number']
            validation_results['harmonic_positive'] = torch.all(harmonic_values > 0.0).item()
        
        if 'mechanics_solution' in predictions:
            kinetic_energy = predictions['mechanics_solution']['kinetic_energy']
            potential_energy = predictions['mechanics_solution']['potential_energy']
            
            validation_results['energies_non_negative'] = (
                torch.all(kinetic_energy >= 0.0).item() and 
                torch.all(potential_energy >= 0.0).item()
            )
        
        return validation_results

# Model factory functions
def create_model(config: EulerNetConfig) -> EulerNet:
    """Factory function to create EulerNet model"""
    model = EulerNet(config)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created EulerNet model:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return model

def load_model(checkpoint_path: str, config: EulerNetConfig) -> EulerNet:
    """Load model from checkpoint"""
    model = create_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded model from {checkpoint_path}")
    
    return model

# Export the main classes and functions
__all__ = [
    'EulerNet',
    'EulerTotientNetwork',
    'EulerPrimeNetwork', 
    'EulerZetaNetwork',
    'EulerMechanicsNetwork',
    'MultiHeadMathematicalAttention',
    'MathematicallyInformedLayer',
    'CrossDomainFusionNetwork',
    'create_model',
    'load_model'
]

if __name__ == "__main__":
    # Test model creation
    from .config import EulerNetConfig
    
    config = EulerNetConfig()
    model = create_model(config)
    
    # Test forward pass with dummy data
    dummy_inputs = {
        'number_theory': torch.randn(4, config.model.number_theory_input_size),
        'analysis': torch.randn(4, config.model.analysis_input_size),
        'mechanics': torch.randn(4, config.model.mechanics_input_size)
    }
    
    with torch.no_grad():
        outputs = model(dummy_inputs)
        validation_results = model.mathematical_validation(dummy_inputs)
        insights = model.get_mathematical_insights(outputs)
    
    print("Model test completed successfully!")
    print("Validation results:", validation_results)
    print("Mathematical insights:", insights)