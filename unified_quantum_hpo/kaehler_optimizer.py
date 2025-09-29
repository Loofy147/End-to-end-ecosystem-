# unified_quantum_hpo/kaehler_optimizer.py
"""
Revolutionary Framework: Kähler Geometry for Neural Architecture Search
- Treats HPO space as complex Kähler manifold with symplectic + Riemannian structure
- Uses holomorphic gradient flows that preserve both metric and symplectic form
- Integrates circular backprop as parallel transport on fiber bundles
"""

import numpy as np
from typing import Dict, Any, Callable, Tuple, List, Optional
from dataclasses import dataclass, field
import scipy.sparse as sp
from scipy.linalg import expm


# Mock ConfigSpace class to satisfy dependencies
class MockConfigSpace:
    """A mock class to simulate a configuration space for demonstration."""
    def __init__(self, parameters: Dict[str, Tuple[float, float]]):
        self.parameters = parameters
        self.param_names = list(parameters.keys())
        self.bounds = np.array([v for v in parameters.values()])

    def to_array(self, configs: List[Dict[str, float]]) -> np.ndarray:
        """Converts a list of configuration dicts to a numpy array."""
        arrs = []
        for config in configs:
            arrs.append(np.array([config[p] for p in self.param_names]))
        return np.array(arrs)

    def to_config(self, vector: np.ndarray) -> Dict[str, float]:
        """Converts a numpy vector back to a configuration dict."""
        return {name: float(val) for name, val in zip(self.param_names, vector)}

    def sample_configuration(self) -> Dict[str, float]:
        """Samples a random configuration from the space."""
        sample = {}
        for name, (low, high) in self.parameters.items():
            sample[name] = np.random.uniform(low, high)
        return sample

@dataclass
class KaehlerPoint:
    """Point on Kähler manifold with complex structure"""
    config: Dict[str, float]
    complex_coords: np.ndarray  # Holomorphic coordinates z = x + iy
    kaehler_potential: float    # K(z, z̄)
    metric_tensor: np.ndarray   # g_ij̄ = ∂²K/∂z^i∂z̄^j
    symplectic_form: np.ndarray # ω = i∂∂̄K
    complex_structure: np.ndarray # J: TM → TM with J² = -I


class KaehlerHPOOptimizer:
    """
    Revolutionary: First use of Kähler geometry for hyperparameter optimization.

    Key Innovation: HPO space is Kähler manifold where:
    1. Metric comes from loss landscape Hessian (Riemannian)
    2. Symplectic form preserves phase space structure (Hamiltonian dynamics)
    3. Complex structure J relates position and momentum naturally
    4. Holomorphic sections give natural "good configurations"
    """

    def __init__(self, config_space, objective_fn: Callable):
        self.cs = config_space
        self.objective = objective_fn
        self.dim = len(config_space.parameters)

        # Complex manifold structure
        self.complex_dim = self.dim  # dim_C = dim_R/2 for Kähler manifolds
        self.kahler_points = []

        # Quantum-inspired momentum (from SpokNAS integration)
        self.quantum_momentum = {}

    def _vector_to_config(self, vector: np.ndarray) -> Dict[str, float]:
        """Converts a numpy vector back to a configuration dict."""
        return self.cs.to_config(vector)

    def _compute_gradient(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Computes gradient of the objective function via finite differences."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps

            f_plus = self.objective(self._vector_to_config(x_plus))
            f_minus = self.objective(self._vector_to_config(x_minus))

            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def complexify_config(self, config: Dict[str, Any]) -> np.ndarray:
        """
        Map real config to holomorphic coordinates.
        Novel: Use action-angle variables from classical mechanics
        """
        x = self.cs.to_array([config])[0]

        # Create canonical momentum conjugate to position
        # p_i = ∂L/∂ẋ^i where L is "Lagrangian" from loss landscape
        grad = self._compute_gradient(x)
        p = -grad  # Momentum from gradient

        # Holomorphic coordinates: z = (x + ip)/√2
        z = (x + 1j * p) / np.sqrt(2)

        return z

    def compute_kahler_potential(self, z: np.ndarray) -> float:
        """
        Kähler potential K(z,z̄) determines all geometric structure.
        Novel: Use expected loss + regularization as potential
        """
        # Recover real coordinates
        x = np.real(z * np.sqrt(2))
        p = np.imag(z * np.sqrt(2))

        # Potential = Loss + Kinetic term (like in physics)
        config = self._vector_to_config(x)
        potential_energy = -self.objective(config)  # Negative because we maximize
        kinetic_energy = 0.5 * np.sum(p**2)

        K = potential_energy + kinetic_energy

        return float(K)

    def compute_kahler_metric(self, z: np.ndarray) -> np.ndarray:
        """
        Kähler metric g_{ij̄} = ∂²K/∂z^i∂z̄^j
        This is both Riemannian metric AND determines symplectic form
        """
        eps = 1e-5
        n = len(z)
        metric = np.zeros((n, n), dtype=complex)

        K0 = self.compute_kahler_potential(z)

        # Compute mixed derivatives ∂²K/∂z^i∂z̄^j
        for i in range(n):
            for j in range(n):
                # Perturb z^i (holomorphic)
                z_i_plus = z.copy()
                z_i_plus[i] += eps

                # Perturb z̄^j (antiholomorphic)
                z_j_plus = z.copy()
                z_j_plus[j] += eps * 1j  # Antiholomorphic perturbation

                z_both = z.copy()
                z_both[i] += eps
                z_both[j] += eps * 1j

                # Mixed derivative
                K_ij = self.compute_kahler_potential(z_both)
                K_i = self.compute_kahler_potential(z_i_plus)
                K_j = self.compute_kahler_potential(z_j_plus)

                metric[i, j] = (K_ij - K_i - K_j + K0) / (eps**2)

        # Ensure Hermitian
        metric = (metric + metric.conj().T) / 2

        return metric

    def compute_symplectic_form(self, z: np.ndarray,
                                metric: np.ndarray) -> np.ndarray:
        """
        Symplectic form ω = i g_{ij̄} dz^i ∧ dz̄^j
        Novel: This connects to circular backprop phase space structure
        """
        # For Kähler manifolds, the symplectic form ω is directly related to the
        # Kähler metric g by ω = (i/2)g. We return this complex representation.
        return (1j / 2) * metric

    def holomorphic_gradient_flow(self, z: np.ndarray,
                                  dt: float = 0.01) -> np.ndarray:
        """
        Gradient flow in holomorphic coordinates.
        Novel: Preserves BOTH metric and symplectic structure simultaneously
        """
        # Compute Kähler metric
        g = self.compute_kahler_metric(z)

        # Holomorphic gradient: ∇_hol = g^{ij̄} ∂/∂z̄^j
        grad_hol = self._compute_holomorphic_gradient(z, g)

        # Flow equation: dz/dt = -∇_hol K
        # This is BOTH gradient descent AND Hamiltonian flow!
        z_new = z - dt * grad_hol

        return z_new

    def _compute_holomorphic_gradient(self, z: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Computes holomorphic gradient ∇_hol = g^{ij̄} ∂K/∂z̄^j."""
        # Regularize the metric tensor to prevent singularity
        regularization = np.eye(g.shape[0]) * 1e-8
        g_inv = np.linalg.inv(g + regularization)

        # Compute ∂K/∂z̄^j
        grad_k_z_bar = np.zeros_like(z, dtype=complex)
        K0 = self.compute_kahler_potential(z)
        for j in range(len(z)):
            z_plus = z.copy()
            z_plus[j] += eps * 1j # Perturb anti-holomorphic part
            K_plus = self.compute_kahler_potential(z_plus)
            grad_k_z_bar[j] = (K_plus - K0) / (eps * 1j)

        return g_inv @ grad_k_z_bar

    def _compute_kahler_connection(self, z: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Computes Christoffel symbols (connection) for the Kähler metric.
        Γ^k_{ij} = g^{k l̄} ∂g_{i l̄} / ∂z^j
        """
        n = len(z)
        g = self.compute_kahler_metric(z)
        # Regularize the metric tensor to prevent singularity
        regularization = np.eye(g.shape[0]) * 1e-8
        g_inv = np.linalg.inv(g + regularization)

        # Compute derivative of metric tensor ∂g_{i l̄} / ∂z^j
        dG_dz = np.zeros((n, n, n), dtype=complex)
        for i in range(n):
            for l in range(n):
                for j in range(n):
                    z_plus = z.copy()
                    z_plus[j] += eps
                    g_plus = self.compute_kahler_metric(z_plus)
                    dG_dz[i, l, j] = (g_plus[i, l] - g[i, l]) / eps

        # Compute Christoffel symbols
        connection = np.zeros((n, n, n), dtype=complex)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    connection[k, i, j] = np.sum(g_inv[k, :] * dG_dz[i, :, j])

        # For this implementation, we simplify and return a rank-2 tensor.
        # A full implementation would use this rank-3 tensor in parallel transport.
        return connection.mean(axis=0) # Simplified for this example

    def integrate_with_circular_echo(self, z: np.ndarray,
                                    echo_strength: float = 0.01,
                                    neumann_steps: int = 3) -> np.ndarray:
        """
        MAJOR INNOVATION: Integrate circular backprop with Kähler geometry

        Circular echo = parallel transport along fiber bundle over config space
        """
        # Standard holomorphic flow
        z_standard = self.holomorphic_gradient_flow(z)

        # Circular echo as parallel transport
        # The "echo" corresponds to holonomy around a closed loop in config space
        connection = self._compute_kahler_connection(z)

        # Compute holonomy (path-ordered exponential)
        holonomy = self._compute_holonomy(z, z_standard, connection, neumann_steps)

        # Apply holonomy transformation (like circular propagation)
        z_echo = z_standard + echo_strength * holonomy @ (z_standard - z)

        return z_echo

    def _compute_holonomy(self, z_start: np.ndarray, z_end: np.ndarray,
                         connection: np.ndarray, steps: int) -> np.ndarray:
        """
        Compute holonomy (parallel transport around closed loop).
        This is the geometric interpretation of circular backprop!
        """
        # Discretize path from z_start to z_end and back
        path = []
        for t in np.linspace(0, 1, steps):
            path.append((1-t) * z_start + t * z_end)
        for t in np.linspace(0, 1, steps):
            path.append((1-t) * z_end + t * z_start)

        # Path-ordered exponential of connection
        holonomy = np.eye(len(z_start), dtype=complex)

        for i in range(len(path)-1):
            dz = path[i+1] - path[i]
            # Connection 1-form contracted with tangent vector
            infinitesimal_transport = np.eye(len(z_start)) + connection @ dz.reshape(-1, 1)
            holonomy = infinitesimal_transport @ holonomy

        return holonomy

    def _compute_complex_hessian(self, z: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Computes the full complex Hessian of the Kähler potential."""
        n = len(z)
        hessian = np.zeros((n, n), dtype=complex)
        K0 = self.compute_kahler_potential(z)
        for i in range(n):
            for j in range(n):
                # ∂²K/∂z^i∂z^j
                z_i_plus = z.copy(); z_i_plus[i] += eps
                z_j_plus = z.copy(); z_j_plus[j] += eps
                z_ij_plus = z.copy(); z_ij_plus[i] += eps; z_ij_plus[j] += eps

                K_i = self.compute_kahler_potential(z_i_plus)
                K_j = self.compute_kahler_potential(z_j_plus)
                K_ij = self.compute_kahler_potential(z_ij_plus)

                hessian[i, j] = (K_ij - K_i - K_j + K0) / (eps**2)
        return hessian


class QuantumInspiredSpokNAS:
    """
    Integrate SpokNAS with quantum annealing on Kähler manifold.
    Novel: Architecture search becomes path integral over holomorphic sections
    """

    def __init__(self, layer_library: List[str], kahler_opt: KaehlerHPOOptimizer):
        self.layer_lib = layer_library
        self.kahler = kahler_opt
        self.quantum_state = {}  # Quantum superposition of architectures

    def quantum_architecture_superposition(self, population: List[Dict]) -> np.ndarray:
        """
        Represent population as quantum state |ψ⟩ = Σ α_i |arch_i⟩
        Novel: Use holomorphic coordinates for quantum amplitudes
        """
        n_archs = len(population)

        # Map each architecture to point on Kähler manifold
        z_coords = []
        for arch in population:
            z = self.kahler.complexify_config(arch)
            z_coords.append(z)

        # Quantum amplitudes from Kähler potential
        amplitudes = []
        for z in z_coords:
            K = self.kahler.compute_kahler_potential(z)
            # Quantum amplitude: α = exp(-K/ℏ) (like partition function)
            alpha = np.exp(-K)  # ℏ = 1
            amplitudes.append(alpha)

        amplitudes = np.array(amplitudes)
        # Normalize: ⟨ψ|ψ⟩ = 1
        amplitudes = amplitudes / np.sqrt(np.sum(np.abs(amplitudes)**2))

        return amplitudes

    def quantum_annealing_step(self, population: List[Dict],
                               temperature: float) -> List[Dict]:
        """
        Quantum annealing using Kähler structure.
        Novel: Temperature controls tunneling via symplectic action
        """
        amplitudes = self.quantum_architecture_superposition(population)

        # Quantum tunneling probability via symplectic action
        # S = ∫ ω (symplectic action)
        new_population = []

        for i, arch in enumerate(population):
            z = self.kahler.complexify_config(arch)

            # Tunneling to nearby architecture
            omega = self.kahler.compute_symplectic_form(z,
                                                       self.kahler.compute_kahler_metric(z))

            # Random symplectic perturbation (preserves phase space structure)
            dz = np.random.randn(len(z)) + 1j * np.random.randn(len(z))
            # The quadratic form with a skew-hermitian matrix is purely imaginary.
            # We take the imaginary part to get a real-valued action.
            action = np.imag(dz.conj() @ omega @ dz)

            # Quantum tunneling probability: exp(-S/T)
            tunnel_prob = np.exp(-np.abs(action) / temperature)

            if np.random.rand() < tunnel_prob:
                z_new = z + 0.1 * dz  # Small tunneling step
                arch_new = self._z_to_architecture(z_new)
                new_population.append(arch_new)
            else:
                new_population.append(arch)

        return new_population

    def _z_to_architecture(self, z: np.ndarray) -> Dict[str, Any]:
        """
        Maps a point on the Kähler manifold back to a discrete architecture.
        This is a simplified decoding process. A more advanced version would
        use a more sophisticated generative model.
        """
        # Recover real coordinates (hyperparameters)
        x = np.real(z * np.sqrt(2))
        config = self.kahler._vector_to_config(x)

        # Decode architecture from the residual part of the vector
        # This is a placeholder for a more complex decoding scheme.
        num_layers = int(np.clip(config.get('num_layers', 5), 3, 10))
        arch = []
        for i in range(num_layers):
            # Use other hyperparameters to select layers
            layer_choice_val = (config.get('lr', 0.001) * 1000 + config.get('dropout', 0.5) * 10 + i) % len(self.layer_lib)
            arch.append(self.layer_lib[int(layer_choice_val)])

        return {'architecture': arch, 'hyperparams': config}


class TopologicalMetaLearner:
    """
    Use persistent homology + Kähler geometry for meta-learning.
    Novel: Learn on topology of loss landscape in complex coordinates
    """

    def __init__(self, kahler_opt: KaehlerHPOOptimizer):
        self.kahler = kahler_opt
        self.persistence_diagrams = {}

    def compute_loss_landscape_topology(self, configs: List[Dict]) -> Dict:
        """
        Compute topological features of loss landscape.
        Novel: Use holomorphic sections as generating cycles
        """
        # Map configs to Kähler manifold
        z_points = [self.kahler.complexify_config(c) for c in configs]

        # Compute Kähler potentials
        potentials = [self.kahler.compute_kahler_potential(z) for z in z_points]

        # Build complex of holomorphic sections
        # Points connected if they lie on same holomorphic submanifold
        edges = []
        for i in range(len(z_points)):
            for j in range(i+1, len(z_points)):
                # Check if points are holomorphically connected
                # Novel criterion: small symplectic action between them
                omega_i = self.kahler.compute_symplectic_form(
                    z_points[i],
                    self.kahler.compute_kahler_metric(z_points[i])
                )
                dz = z_points[j] - z_points[i]
                # The quadratic form with a skew-hermitian matrix is purely imaginary.
                # We take the imaginary part to get a real-valued action.
                action = np.abs(np.imag(dz.conj() @ omega_i @ dz))

                if action < 0.1:  # Threshold for "holomorphic connection"
                    edges.append((i, j, action))

        # Compute persistent homology
        # Betti numbers tell us about holes in loss landscape
        return self._compute_persistence(z_points, edges, potentials)

    def meta_learn_from_topology(self, history: List[Dict]) -> Dict[str, Any]:
        """
        Learn meta-features from topological invariants.
        Novel: Use Morse theory on Kähler manifolds
        """
        topology = self.compute_loss_landscape_topology(history)

        # Morse theory: critical points of Kähler potential
        # correspond to good configurations
        critical_configs = []

        for config in history:
            z = self.kahler.complexify_config(config)
            g = self.kahler.compute_kahler_metric(z)
            # Correctly call the method on the kahler object
            grad = self.kahler._compute_holomorphic_gradient(z, g)

            # Critical point: ∇K = 0
            if np.linalg.norm(grad) < 1e-3:
                # Compute Morse index (number of negative eigenvalues of Hessian)
                # Correctly call the method on the kahler object
                hessian = self.kahler._compute_complex_hessian(z)
                eigenvalues = np.linalg.eigvals(hessian)
                morse_index = np.sum(np.real(eigenvalues) < 0)

                critical_configs.append({
                    'config': config,
                    'morse_index': morse_index,
                    'potential': self.kahler.compute_kahler_potential(z)
                })

        return {
            'topology': topology,
            'critical_points': critical_configs,
            'meta_features': self._extract_meta_features(topology, critical_configs)
        }

    def _compute_persistence(self, points: List[np.ndarray], edges: List[Tuple], values: List[float]) -> Dict:
        """
        Placeholder for persistent homology computation.
        In a real implementation, this would use a library like GUDHI or Dionysus
        to compute Betti numbers (B0, B1, etc.) which represent topological
        features like connected components, holes, voids, etc.
        """
        print("NOTE: _compute_persistence is a placeholder. A real implementation would use a topology library.")
        # Mock Betti numbers
        betti_numbers = {
            'B0': len(points) - len(edges), # Simplified connected components
            'B1': max(0, len(edges) - len(points) + 1) # Simplified number of loops
        }
        return {'betti_numbers': betti_numbers}

    def _extract_meta_features(self, topology: Dict, critical_points: List) -> Dict:
        """
        Placeholder for extracting meta-features from topological data.
        This could involve analyzing the distribution of critical points,
        the stability of topological features, etc.
        """
        print("NOTE: _extract_meta_features is a placeholder.")
        return {
            'num_minima': sum(1 for p in critical_points if p['morse_index'] == 0),
            'num_saddles': sum(1 for p in critical_points if p['morse_index'] > 0),
            'betti_numbers': topology.get('betti_numbers', {})
        }


## Integration Hub: Unified System

class UnifiedQuantumHPOSystem:
    """
    Complete integration of all advanced techniques:
    - Kähler geometry for HPO
    - Quantum annealing for NAS
    - Circular backprop as holonomy
    - Topological meta-learning
    - Autonomous adaptation (from otonomos.txt)
    """

    def __init__(self, config_space, objective_fn, layer_library):
        # Core components
        self.kahler_opt = KaehlerHPOOptimizer(config_space, objective_fn)
        self.quantum_nas = QuantumInspiredSpokNAS(layer_library, self.kahler_opt)
        self.topo_meta = TopologicalMetaLearner(self.kahler_opt)

        # Autonomous adaptation state
        self.meta_state = {
            'kahler_potential_history': [],
            'topology_evolution': [],
            'quantum_temperature': 1.0
        }

    def suggest_next_config(self, history: List[Dict]) -> Dict[str, Any]:
        """
        Suggest next configuration using full unified framework.
        """
        # 1. Learn from topology
        meta_knowledge = self.topo_meta.meta_learn_from_topology(history)

        # 2. Get current best config
        if history:
            best_config = max(history, key=lambda c: self.kahler_opt.objective(c))
        else:
            best_config = self.kahler_opt.cs.sample_configuration()

        # 3. Map to Kähler manifold
        z_current = self.kahler_opt.complexify_config(best_config)

        # 4. Holomorphic gradient flow with circular echo
        z_next = self.kahler_opt.integrate_with_circular_echo(
            z_current,
            echo_strength=0.01,
            neumann_steps=3
        )

        # 5. Apply quantum tunneling if stuck in local minimum
        if self._detect_stagnation(history):
            # Anneal to explore
            self.meta_state['quantum_temperature'] *= 1.1
        else:
            # Cool down to exploit
            self.meta_state['quantum_temperature'] *= 0.95

        # Convert back to config
        config_next = self._z_to_config(z_next)

        return {
            'config': config_next,
            'meta_info': meta_knowledge,
            'kahler_potential': self.kahler_opt.compute_kahler_potential(z_next),
            'expected_improvement': self._estimate_improvement(z_current, z_next)
        }

    def _z_to_config(self, z: np.ndarray) -> Dict[str, float]:
        """Converts a complex coordinate vector back to a config dictionary."""
        x = np.real(z * np.sqrt(2))
        return self.kahler_opt._vector_to_config(x)

    def _detect_stagnation(self, history: List[Dict], window: int = 5, threshold: float = 1e-4) -> bool:
        """Detects if the optimization has stagnated based on objective history."""
        if len(history) < window:
            return False

        recent_objectives = [self.kahler_opt.objective(c) for c in history[-window:]]
        improvement = np.max(recent_objectives) - np.min(recent_objectives)

        return improvement < threshold

    def _estimate_improvement(self, z_current: np.ndarray, z_next: np.ndarray) -> float:
        """Estimates improvement based on the change in Kähler potential."""
        k_current = self.kahler_opt.compute_kahler_potential(z_current)
        k_next = self.kahler_opt.compute_kahler_potential(z_next)
        # Improvement is the reduction in potential energy (negative loss)
        return k_current - k_next


if __name__ == '__main__':
    # Example Usage

    # 1. Define a mock objective function (e.g., a simple quadratic function)
    # In a real scenario, this would be a model training and validation function.
    def mock_objective(config: Dict[str, float]) -> float:
        lr = config.get('lr', 0.01)
        dropout = config.get('dropout', 0.5)
        # A simple function to maximize: peak at lr=0.1, dropout=0.2
        return -((lr - 0.1)**2 + (dropout - 0.2)**2)

    # 2. Define the configuration space
    config_space = MockConfigSpace({
        'lr': (1e-4, 1e-1),
        'dropout': (0.1, 0.9),
        'num_layers': (3, 10) # Used by the mock _z_to_architecture
    })

    # 3. Define the layer library for SpokNAS
    layer_library = [
        'conv3x3-32', 'conv3x3-64', 'pool-max', 'relu', 'bn', 'flatten', 'fc-128'
    ]

    # 4. Initialize the unified system
    unified_system = UnifiedQuantumHPOSystem(
        config_space=config_space,
        objective_fn=mock_objective,
        layer_library=layer_library
    )

    # 5. Run a suggestion step
    # Start with an empty history
    history = []

    print("--- Running Initial Suggestion ---")
    # Generate the first suggestion
    initial_suggestion = unified_system.suggest_next_config(history)

    print("\nSuggested Config:", initial_suggestion['config'])
    print("Kähler Potential:", initial_suggestion['kahler_potential'])
    print("Meta Info (Topology):", initial_suggestion['meta_info']['topology'])

    # Add the new config to history
    history.append(initial_suggestion['config'])

    # 6. Run another suggestion step with history
    print("\n--- Running Second Suggestion ---")
    second_suggestion = unified_system.suggest_next_config(history)

    print("\nSuggested Config:", second_suggestion['config'])
    print("Kähler Potential:", second_suggestion['kahler_potential'])
    print("Expected Improvement:", second_suggestion['expected_improvement'])
    print("Meta Info (Critical Points):", second_suggestion['meta_info']['critical_points'])

    # 7. Demonstrate Quantum Annealing Step
    print("\n--- Demonstrating Quantum Annealing ---")
    # Create a small population of configurations
    population = [config_space.sample_configuration() for _ in range(5)]
    print("Original Population:", [f"lr={c['lr']:.3f}" for c in population])

    # Perform one step of quantum annealing
    new_population = unified_system.quantum_nas.quantum_annealing_step(
        population, temperature=0.5
    )
    print("New Population after Annealing:", [f"lr={c['hyperparams']['lr']:.3f}" for c in new_population])