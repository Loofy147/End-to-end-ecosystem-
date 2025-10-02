







import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, defaultdict
import math

class HierarchicalState:
    """
    Multi-scale state representation that works across different target sizes.
    Combines local and global features for better long-term planning.
    """
    def __init__(self, current, target, step, max_steps, forbidden_states=None):
        self.current = current
        self.target = target
        self.step = step
        self.max_steps = max_steps
        self.forbidden_states = forbidden_states or set()
    
    def to_features(self):
        """Multi-scale feature representation"""
        if self.target == 0:
            return np.zeros(12)
        
        # Scale-invariant features
        progress_ratio = self.current / self.target
        remaining_ratio = (self.target - self.current) / self.target
        time_ratio = self.step / self.max_steps
        
        # Multi-scale gap analysis
        gap = abs(self.target - self.current)
        log_gap = math.log(gap + 1) / math.log(self.target + 1)  # Logarithmic scale
        gap_magnitude = gap / self.target  # Relative scale
        
        # Strategic features
        is_close = 1.0 if gap <= 10 else 0.0  # Near-target flag
        is_far = 1.0 if gap >= self.target * 0.5 else 0.0  # Far-target flag
        
        # Constraint features
        danger_proximity = self._compute_danger_proximity()
        constraint_pressure = self._compute_constraint_pressure()
        
        # Phase identification
        phase = self._identify_phase()
        
        # Efficiency features
        theoretical_min_steps = self._compute_min_steps()
        efficiency_ratio = theoretical_min_steps / (self.max_steps - self.step + 1)
        
        return np.array([
            progress_ratio, remaining_ratio, time_ratio,
            log_gap, gap_magnitude, is_close, is_far,
            danger_proximity, constraint_pressure,
            phase, theoretical_min_steps, efficiency_ratio
        ])
    
    def _compute_danger_proximity(self):
        """Distance to nearest forbidden state"""
        if not self.forbidden_states:
            return 0.0
        
        distances = [abs(self.current - forbidden) for forbidden in self.forbidden_states]
        min_distance = min(distances)
        return 1.0 / (min_distance + 1)  # Closer = higher value
    
    def _compute_constraint_pressure(self):
        """How constrained is the current position"""
        if not self.forbidden_states:
            return 0.0
        
        # Count forbidden states in the path to target
        if self.current < self.target:
            path_range = range(self.current + 1, self.target + 1)
        else:
            path_range = range(self.target, self.current)
        
        blocked_path_states = sum(1 for state in path_range if state in self.forbidden_states)
        path_length = len(path_range)
        
        return blocked_path_states / (path_length + 1) if path_length > 0 else 0.0
    
    def _identify_phase(self):
        """Identify problem-solving phase: exploration(0), navigation(1), precision(2)"""
        gap = abs(self.target - self.current)
        
        if gap > self.target * 0.7:
            return 0.0  # Exploration phase
        elif gap > 10:
            return 1.0  # Navigation phase
        else:
            return 2.0  # Precision phase
    
    def _compute_min_steps(self):
        """Theoretical minimum steps to target"""
        gap = abs(self.target - self.current)
        # Assuming max action is 5
        return math.ceil(gap / 5.0)

class HierarchicalQNetwork(nn.Module):
    """
    Multi-head network that learns different strategies for different phases
    """
    def __init__(self, state_dim=12, action_dim=3, hidden_dim=256):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Phase-specific heads
        self.exploration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.navigation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.precision_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Phase classifier
        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state_features):
        # Extract shared features
        features = self.feature_extractor(state_features)
        
        # Classify phase
        phase_probs = self.phase_classifier(features)
        
        # Compute Q-values for each phase
        exploration_q = self.exploration_head(features)
        navigation_q = self.navigation_head(features)
        precision_q = self.precision_head(features)
        
        # Weighted combination based on phase probabilities
        q_values = (phase_probs[:, 0:1] * exploration_q + 
                   phase_probs[:, 1:2] * navigation_q + 
                   phase_probs[:, 2:3] * precision_q)
        
        return q_values, phase_probs

class GoalDecompositionAgent:
    """
    Agent that decomposes large targets into manageable sub-goals
    """
    def __init__(self, actions=[1, 3, 5], lr=0.0005):
        self.actions = actions
        self.max_action = max(actions)
        
        # Networks
        self.q_network = HierarchicalQNetwork()
        self.target_network = HierarchicalQNetwork()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.experience_buffer = deque(maxlen=10000)
        
        # Goal decomposition
        self.subgoal_stack = []
        self.current_subgoal = None
        
        # Adaptive exploration
        self.epsilon_schedule = self._create_epsilon_schedule()
        self.step_count = 0
        
    def _create_epsilon_schedule(self):
        """Adaptive epsilon that increases for harder problems"""
        def epsilon_fn(step, target_difficulty):
            base_epsilon = 0.1
            difficulty_bonus = min(0.2, target_difficulty / 1000)
            decay = max(0.01, base_epsilon * (0.995 ** (step / 100)))
            return decay + difficulty_bonus
        return epsilon_fn
    
    def decompose_target(self, current, target):
        """Decompose large targets into manageable sub-goals"""
        gap = abs(target - current)
        
        if gap <= 50:
            # Direct approach for smaller gaps
            return [target]
        
        # Create intermediate sub-goals
        subgoals = []
        direction = 1 if target > current else -1
        
        # Strategic waypoints
        num_subgoals = max(2, gap // 75)  # One subgoal per ~75 units
        step_size = gap // num_subgoals
        
        for i in range(1, num_subgoals):
            subgoal = current + (step_size * i * direction)
            subgoals.append(subgoal)
        
        subgoals.append(target)  # Final target
        return subgoals
    
    def get_state_features(self, current, target, step, max_steps, forbidden_states=None):
        """Enhanced state representation"""
        hierarchical_state = HierarchicalState(current, target, step, max_steps, forbidden_states)
        return hierarchical_state.to_features()
    
    def choose_action(self, current, target, step, max_steps, forbidden_states=None, training=True):
        """Choose action with hierarchical reasoning"""
        # Update current subgoal if needed
        if not self.subgoal_stack:
            self.subgoal_stack = self.decompose_target(current, target)
            self.current_subgoal = self.subgoal_stack.pop(0)
        
        # Check if current subgoal is reached
        if current == self.current_subgoal and self.subgoal_stack:
            self.current_subgoal = self.subgoal_stack.pop(0)
            print(f"Subgoal reached! Next subgoal: {self.current_subgoal}")
        
        # Use current subgoal for decision making
        working_target = self.current_subgoal if self.current_subgoal else target
        
        # Adaptive exploration
        difficulty = abs(target - current)
        epsilon = self.epsilon_schedule(self.step_count, difficulty)
        
        if training and random.random() < epsilon:
            return random.choice(self.actions)
        
        # Neural network decision
        state_features = self.get_state_features(current, working_target, step, max_steps, forbidden_states)
        state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
        
        with torch.no_grad():
            q_values, phase_probs = self.q_network(state_tensor)
        
        # Constraint checking
        valid_actions = []
        for i, action in enumerate(self.actions):
            next_state = current + action
            if forbidden_states and next_state in forbidden_states:
                continue  # Skip forbidden states
            valid_actions.append((i, action))
        
        if not valid_actions:
            # Emergency backtracking
            return -self.actions[0] if -self.actions[0] in self.actions else self.actions[0]
        
        # Choose best valid action
        best_idx = -1
        best_value = float('-inf')
        
        for idx, action in valid_actions:
            if q_values[0, idx] > best_value:
                best_value = q_values[0, idx]
                best_idx = idx
        
        return self.actions[best_idx]
    
    def store_experience(self, current, target, step, max_steps, action, reward, 
                        next_current, next_step, done, forbidden_states=None):
        """Store experience with enhanced state representation"""
        experience = {
            'state': self.get_state_features(current, target, step, max_steps, forbidden_states),
            'action': self.actions.index(action) if action in self.actions else 0,
            'reward': reward,
            'next_state': self.get_state_features(next_current, target, next_step, max_steps, forbidden_states),
            'done': done
        }
        self.experience_buffer.append(experience)
    
    def learn_from_experience(self, batch_size=64):
        """Enhanced learning with prioritized experience replay"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.experience_buffer, batch_size)
        
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch])
        
        # Current Q-values
        current_q_values, _ = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Loss computation
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % 200 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def reset_subgoals(self):
        """Reset subgoal tracking for new episode"""
        self.subgoal_stack = []
        self.current_subgoal = None

def test_scaling_agent(agent, test_targets, max_steps_mult=1.5):
    """Test the hierarchical agent on large targets"""
    print("=== SCALING TEST WITH HIERARCHICAL AGENT ===")
    
    results = {}
    
    for target in test_targets:
        print(f"\nTesting target {target}:")
        
        # Dynamic step limit based on target size
        theoretical_min = math.ceil(target / 5)
        max_steps = int(theoretical_min * max_steps_mult)
        
        print(f"  Theoretical minimum: {theoretical_min} steps")
        print(f"  Allowed steps: {max_steps}")
        
        # Reset agent state
        agent.reset_subgoals()
        
        # Run episode
        current = 0
        step = 0
        path = [current]
        subgoals_reached = []
        
        while step < max_steps and current != target:
            action = agent.choose_action(current, target, step, max_steps, training=False)
            current += action
            step += 1
            path.append(current)
            
            # Track subgoal progress
            if hasattr(agent, 'current_subgoal') and current == agent.current_subgoal:
                subgoals_reached.append(current)
        
        success = current == target
        efficiency = (theoretical_min / step * 100) if success else 0
        
        results[target] = {
            'success': success,
            'steps': step,
            'efficiency': efficiency,
            'path': path[:10] + ['...'] + path[-5:] if len(path) > 15 else path,
            'subgoals_reached': subgoals_reached
        }
        
        print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
        print(f"  Steps taken: {step}")
        print(f"  Efficiency: {efficiency:.1f}%")
        print(f"  Subgoals reached: {subgoals_reached}")
        if len(path) <= 15:
            print(f"  Full path: {' → '.join(map(str, path))}")
        else:
            print(f"  Path: {' → '.join(map(str, path[:5]))} ... {' → '.join(map(str, path[-5:]))}")
    
    return results

# Usage example
if __name__ == "__main__":
    # Create hierarchical agent
    agent = GoalDecompositionAgent()
    
    # Quick training on medium-scale problems to establish basic patterns
    print("Quick bootstrap training...")
    for _ in range(1000):
        target = random.randint(20, 100)
        current = 0
        step = 0
        max_steps = int(target / 5 * 1.5)
        
        while step < max_steps and current != target:
            action = agent.choose_action(current, target, step, max_steps)
            next_current = current + action
            
            # Reward shaping
            if next_current == target:
                reward = 100
            elif abs(next_current - target) < abs(current - target):
                reward = 10
            else:
                reward = -1
            
            done = next_current == target or step + 1 >= max_steps
            agent.store_experience(current, target, step, max_steps, action, 
                                 reward, next_current, step + 1, done)
            
            current = next_current
            step += 1
        
        if len(agent.experience_buffer) >= 64:
            agent.learn_from_experience()
    
    # Test on large targets
    large_targets = [150, 200, 300, 500, 750, 1000]
    results = test_scaling_agent(agent, large_targets)
    
    # Analysis
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\n=== FINAL RESULTS ===")
    print(f"Success rate: {successful}/{len(large_targets)} ({successful/len(large_targets)*100:.1f}%)")
    
    if successful > 0:
        avg_efficiency = np.mean([r['efficiency'] for r in results.values() if r['success']])
        print(f"Average efficiency: {avg_efficiency:.1f}%")





