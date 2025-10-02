import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import math
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HierarchicalQNetwork(nn.Module):
    """Enhanced Q-Network with hierarchical processing for relational states"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 128, 64]):
        super(HierarchicalQNetwork, self).__init__()
        
        # Relational feature processing layers
        self.relational_processor = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Hierarchical processing
        self.strategic_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.tactical_layer = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.ReLU()
        )
        
        # Output layers
        self.value_head = nn.Linear(hidden_dims[2], 1)
        self.advantage_head = nn.Linear(hidden_dims[2], action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        # Process relational features
        relational_features = self.relational_processor(state)
        
        # Hierarchical processing
        strategic_features = self.strategic_layer(relational_features)
        tactical_features = self.tactical_layer(strategic_features)
        
        # Dueling DQN architecture
        value = self.value_head(tactical_features)
        advantage = self.advantage_head(tactical_features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class Actor(nn.Module):
    """Enhanced Actor network for continuous action spaces"""
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        # Relational state processor
        self.relational_processor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Policy layers
        self.policy_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        relational_features = self.relational_processor(state)
        return self.max_action * self.policy_net(relational_features)

class Critic(nn.Module):
    """Enhanced Critic network with relational state processing"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()
        
        # State processing
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Combined state-action processing
        self.q_net = nn.Sequential(
            nn.Linear(128 + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state, action):
        state_features = self.state_processor(state)
        combined = torch.cat([state_features, action], dim=1)
        return self.q_net(combined)

class MultiAgentEnvironment:
    """Enhanced multi-agent environment with relational state features"""
    
    def __init__(self, num_agents: int = 2, grid_size: int = 10):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.agents_positions = np.random.rand(num_agents, 2) * grid_size
        self.targets = np.random.rand(num_agents, 2) * grid_size
        self.obstacles = self._generate_obstacles()
        self.step_count = 0
        self.max_steps = 1000
        
        # Game theory parameters
        self.cooperation_bonus = 1.0
        self.competition_penalty = -0.5
        self.collision_penalty = -2.0
    
    def _generate_obstacles(self, num_obstacles: int = 5) -> List[Dict]:
        """Generate random obstacles in the environment"""
        obstacles = []
        for _ in range(num_obstacles):
            obstacle = {
                'center': np.random.rand(2) * self.grid_size,
                'radius': np.random.uniform(0.5, 1.5)
            }
            obstacles.append(obstacle)
        return obstacles
    
    def reset(self):
        """Reset the environment"""
        self.agents_positions = np.random.rand(self.num_agents, 2) * self.grid_size
        self.targets = np.random.rand(self.num_agents, 2) * self.grid_size
        self.step_count = 0
        return self._get_observations()
    
    def step(self, actions: List[np.ndarray]):
        """Execute actions and return next states, rewards, done flags"""
        # Update positions based on actions
        for i, action in enumerate(actions):
            new_pos = self.agents_positions[i] + action * 0.1  # Scale factor
            new_pos = np.clip(new_pos, 0, self.grid_size)  # Keep in bounds
            
            # Check obstacle collisions
            if not self._check_collision(new_pos):
                self.agents_positions[i] = new_pos
        
        # Calculate rewards
        rewards = [self._calculate_reward(i) for i in range(self.num_agents)]
        
        # Check if done
        self.step_count += 1
        done = self.step_count >= self.max_steps or all(self._agent_reached_target(i) for i in range(self.num_agents))
        
        return self._get_observations(), rewards, done, {}
    
    def _check_collision(self, position: np.ndarray) -> bool:
        """Check if position collides with obstacles"""
        for obstacle in self.obstacles:
            distance = np.linalg.norm(position - obstacle['center'])
            if distance < obstacle['radius']:
                return True
        return False
    
    def _agent_reached_target(self, agent_id: int) -> bool:
        """Check if agent reached its target"""
        distance = np.linalg.norm(self.agents_positions[agent_id] - self.targets[agent_id])
        return distance < 0.5
    
    def _calculate_reward(self, agent_id: int) -> float:
        """Enhanced reward function with game theory elements"""
        agent_pos = self.agents_positions[agent_id]
        target_pos = self.targets[agent_id]
        
        # Base reward: negative distance to target
        distance_to_target = np.linalg.norm(agent_pos - target_pos)
        base_reward = -distance_to_target / self.grid_size
        
        # Target reached bonus
        target_bonus = 10.0 if self._agent_reached_target(agent_id) else 0.0
        
        # Collision penalty
        collision_penalty = self.collision_penalty if self._check_collision(agent_pos) else 0.0
        
        # Game theory rewards
        cooperation_reward = self._calculate_cooperation_reward(agent_id)
        competition_reward = self._calculate_competition_reward(agent_id)
        
        # Efficiency bonus
        efficiency_bonus = self._calculate_efficiency_bonus(agent_id)
        
        total_reward = (base_reward + target_bonus + collision_penalty + 
                       cooperation_reward + competition_reward + efficiency_bonus)
        
        return total_reward
    
    def _calculate_cooperation_reward(self, agent_id: int) -> float:
        """Calculate cooperation reward based on helping other agents"""
        cooperation_reward = 0.0
        agent_pos = self.agents_positions[agent_id]
        
        for other_id in range(self.num_agents):
            if other_id == agent_id:
                continue
            
            other_pos = self.agents_positions[other_id]
            other_target = self.targets[other_id]
            
            # Reward for being close to other agents (cooperation)
            distance_to_other = np.linalg.norm(agent_pos - other_pos)
            if distance_to_other < 2.0:  # Cooperation threshold
                cooperation_reward += self.cooperation_bonus * (2.0 - distance_to_other) / 2.0
        
        return cooperation_reward
    
    def _calculate_competition_reward(self, agent_id: int) -> float:
        """Calculate competition penalty for conflicting goals"""
        competition_penalty = 0.0
        agent_pos = self.agents_positions[agent_id]
        
        for other_id in range(self.num_agents):
            if other_id == agent_id:
                continue
            
            other_pos = self.agents_positions[other_id]
            
            # Penalty for being too close (competition/collision)
            distance_to_other = np.linalg.norm(agent_pos - other_pos)
            if distance_to_other < 1.0:  # Competition threshold
                competition_penalty += self.competition_penalty
        
        return competition_penalty
    
    def _calculate_efficiency_bonus(self, agent_id: int) -> float:
        """Calculate efficiency bonus based on path optimality"""
        if self.step_count == 0:
            return 0.0
        
        # Simple efficiency metric: reward for making progress
        current_distance = np.linalg.norm(self.agents_positions[agent_id] - self.targets[agent_id])
        theoretical_min_steps = current_distance / 0.1  # Based on max step size
        actual_steps = self.step_count
        
        if actual_steps > 0:
            efficiency_ratio = min(1.0, theoretical_min_steps / actual_steps)
            return efficiency_ratio * 0.5
        
        return 0.0
    
    def _get_observations(self) -> List[np.ndarray]:
        """Get observations for all agents using relational state representation"""
        observations = []
        for i in range(self.num_agents):
            obs = self._get_agent_observation(i)
            observations.append(obs)
        return observations
    
    def _get_agent_observation(self, agent_id: int) -> np.ndarray:
        """Get relational state observation for a specific agent"""
        agent_pos = self.agents_positions[agent_id]
        target_pos = self.targets[agent_id]
        
        # Initialize observation components
        obs_components = []
        
        # 1. Progress and remaining ratios (2 features)
        total_distance = np.linalg.norm(agent_pos - target_pos) + 1e-8  # Avoid division by zero
        initial_distance = self.grid_size * np.sqrt(2)  # Maximum possible distance
        progress_ratio = max(0, 1 - total_distance / initial_distance)
        remaining_ratio = total_distance / initial_distance
        
        obs_components.extend([progress_ratio, remaining_ratio])
        
        # 2. Multi-scale gap analysis (3 features)
        linear_gap_ratio = total_distance / initial_distance
        log_gap_ratio = np.log(1 + total_distance) / np.log(1 + initial_distance)
        sqrt_gap_ratio = np.sqrt(total_distance) / np.sqrt(initial_distance)
        
        obs_components.extend([linear_gap_ratio, log_gap_ratio, sqrt_gap_ratio])
        
        # 3. Time analysis (3 features)
        time_progress_ratio = min(1.0, self.step_count / self.max_steps)
        estimated_remaining_time = total_distance / 0.1  # Based on max step size
        time_efficiency_ratio = min(1.0, estimated_remaining_time / max(1, self.step_count))
        
        obs_components.extend([time_progress_ratio, estimated_remaining_time / self.max_steps, time_efficiency_ratio])
        
        # 4. Constraint features (4 features)
        constraint_pressure = self._compute_constraint_pressure(agent_pos)
        forbidden_proximity = self._compute_forbidden_proximity(agent_pos)
        trap_pressure = self._compute_trap_pressure(agent_pos)
        escape_routes = self._compute_escape_routes(agent_pos)
        
        obs_components.extend([constraint_pressure, forbidden_proximity, trap_pressure, escape_routes])
        
        # 5. Phase identification (4 features)
        phase_info = self._identify_phase(progress_ratio, constraint_pressure)
        obs_components.extend(phase_info)
        
        # 6. Efficiency metrics (2 features)
        efficiency_ratio = min(1.0, estimated_remaining_time / max(1, self.step_count))
        waste_ratio = max(0, (self.step_count - estimated_remaining_time) / max(1, estimated_remaining_time))
        
        obs_components.extend([efficiency_ratio, min(1.0, waste_ratio)])
        
        # 7. Game theory features (4 features)
        cooperation_potential = self._compute_cooperation_potential(agent_id)
        competition_pressure = self._compute_competition_pressure(agent_id)
        social_distance_ratio = self._compute_social_distance_ratio(agent_id)
        collective_progress = self._compute_collective_progress()
        
        obs_components.extend([cooperation_potential, competition_pressure, social_distance_ratio, collective_progress])
        
        # 8. Directional features (4 features)
        direction_to_target = target_pos - agent_pos
        direction_norm = np.linalg.norm(direction_to_target)
        if direction_norm > 0:
            direction_to_target = direction_to_target / direction_norm
        else:
            direction_to_target = np.array([0.0, 0.0])
        
        # Add directional features
        obs_components.extend([direction_to_target[0], direction_to_target[1]])
        
        # Relative position to environment center
        center_pos = np.array([self.grid_size / 2, self.grid_size / 2])
        relative_to_center = (agent_pos - center_pos) / (self.grid_size / 2)
        obs_components.extend(relative_to_center.tolist())
        
        # Ensure we have exactly 32 features
        while len(obs_components) < 32:
            obs_components.append(0.0)
        
        return np.array(obs_components[:32], dtype=np.float32)
    
    def _compute_constraint_pressure(self, position: np.ndarray) -> float:
        """Compute constraint pressure from obstacles"""
        pressure = 0.0
        for obstacle in self.obstacles:
            distance = np.linalg.norm(position - obstacle['center'])
            safe_distance = obstacle['radius'] + 0.5
            if distance < safe_distance:
                pressure += np.exp(-(distance - obstacle['radius']) / 0.5)
        return min(1.0, np.tanh(pressure))
    
    def _compute_forbidden_proximity(self, position: np.ndarray) -> float:
        """Compute proximity to forbidden areas"""
        if not self.obstacles:
            return 0.0
        
        min_distance = min(max(0, np.linalg.norm(position - obs['center']) - obs['radius']) 
                          for obs in self.obstacles)
        return np.exp(-min_distance / 2.0)
    
    def _compute_trap_pressure(self, position: np.ndarray) -> float:
        """Compute trap pressure (how surrounded the agent is)"""
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        blocked_directions = 0
        
        for dx, dy in directions:
            test_pos = position + np.array([dx, dy]) * 0.5
            # Check bounds
            if (test_pos[0] < 0 or test_pos[0] > self.grid_size or 
                test_pos[1] < 0 or test_pos[1] > self.grid_size):
                blocked_directions += 1
                continue
            # Check obstacles
            if self._check_collision(test_pos):
                blocked_directions += 1
        
        return blocked_directions / len(directions)
    
    def _compute_escape_routes(self, position: np.ndarray) -> float:
        """Compute number of available escape routes"""
        return 1.0 - self._compute_trap_pressure(position)
    
    def _identify_phase(self, progress_ratio: float, constraint_pressure: float) -> List[float]:
        """Identify current phase with adaptive thresholds"""
        # Adaptive thresholds based on constraint pressure
        early_threshold = 0.25 + constraint_pressure * 0.1
        end_threshold = 0.75 - constraint_pressure * 0.05
        
        if progress_ratio < early_threshold:
            phase = [1, 0, 0]  # early
            phase_progress = progress_ratio / early_threshold
        elif progress_ratio < end_threshold:
            phase = [0, 1, 0]  # mid
            phase_progress = (progress_ratio - early_threshold) / (end_threshold - early_threshold)
        else:
            phase = [0, 0, 1]  # end
            phase_progress = (progress_ratio - end_threshold) / (1 - end_threshold)
        
        return phase + [phase_progress]
    
    def _compute_cooperation_potential(self, agent_id: int) -> float:
        """Compute potential for cooperation with other agents"""
        if self.num_agents == 1:
            return 0.0
        
        agent_pos = self.agents_positions[agent_id]
        cooperation_score = 0.0
        
        for other_id in range(self.num_agents):
            if other_id == agent_id:
                continue
            
            other_pos = self.agents_positions[other_id]
            distance = np.linalg.norm(agent_pos - other_pos)
            
            # Higher cooperation potential when agents are close but not too close
            if 1.0 < distance < 3.0:
                cooperation_score += (3.0 - distance) / 2.0
        
        return min(1.0, cooperation_score / (self.num_agents - 1))
    
    def _compute_competition_pressure(self, agent_id: int) -> float:
        """Compute competition pressure from other agents"""
        if self.num_agents == 1:
            return 0.0
        
        agent_pos = self.agents_positions[agent_id]
        competition_score = 0.0
        
        for other_id in range(self.num_agents):
            if other_id == agent_id:
                continue
            
            other_pos = self.agents_positions[other_id]
            distance = np.linalg.norm(agent_pos - other_pos)
            
            # Higher competition when agents are very close
            if distance < 1.5:
                competition_score += (1.5 - distance) / 1.5
        
        return min(1.0, competition_score)
    
    def _compute_social_distance_ratio(self, agent_id: int) -> float:
        """Compute social distance ratio to other agents"""
        if self.num_agents == 1:
            return 1.0
        
        agent_pos = self.agents_positions[agent_id]
        distances = []
        
        for other_id in range(self.num_agents):
            if other_id == agent_id:
                continue
            other_pos = self.agents_positions[other_id]
            distances.append(np.linalg.norm(agent_pos - other_pos))
        
        avg_distance = np.mean(distances)
        max_possible_distance = self.grid_size * np.sqrt(2)
        
        return min(1.0, avg_distance / max_possible_distance)
    
    def _compute_collective_progress(self) -> float:
        """Compute collective progress of all agents"""
        total_progress = 0.0
        
        for i in range(self.num_agents):
            agent_pos = self.agents_positions[i]
            target_pos = self.targets[i]
            distance = np.linalg.norm(agent_pos - target_pos)
            initial_distance = self.grid_size * np.sqrt(2)
            progress = max(0, 1 - distance / initial_distance)
            total_progress += progress
        
        return total_progress / self.num_agents

class GameTheoryAgent:
    """Enhanced Game Theory Agent with Relational State Features"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 agent_id: int,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 device: str = 'cpu'):
        
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Experience replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Hierarchical goal management
        self.subgoals = []
        self.current_subgoal_index = 0
        self.subgoal_threshold = 0.5
        
        # Performance tracking
        self.episode_rewards = []
        self.cooperation_history = []
        self.competition_history = []
        
        # Relational state tracking
        self.previous_state = None
        self.movement_history = deque(maxlen=20)
        
        # Constraint awareness parameters
        self.constraint_threshold = 0.3
        self.safety_margin = 0.1
        
    def compute_relational_state(self, raw_observation: np.ndarray, environment: MultiAgentEnvironment) -> np.ndarray:
        """
        Enhanced relational state computation incorporating all principles:
        - Progress ratio and remaining ratio
        - Multi-scale gap analysis (linear, log, sqrt)
        - Constraint features and trap pressure
        - Phase identification with adaptive thresholds
        - Efficiency metrics
        - Game theory features
        """
        # The environment already computes relational features, but we can enhance them here
        enhanced_state = raw_observation.copy()
        
        # Add temporal consistency features
        if self.previous_state is not None:
            # State change velocity
            state_velocity = np.linalg.norm(enhanced_state[:2] - self.previous_state[:2])
            # Progress rate
            progress_rate = enhanced_state[0] - self.previous_state[0]  # Progress ratio change
            
            # Append temporal features if we have room (pad or replace last features)
            if len(enhanced_state) >= 30:
                enhanced_state[-2] = state_velocity
                enhanced_state[-1] = progress_rate
        
        # Store current state for next iteration
        self.previous_state = enhanced_state.copy()
        
        # Record movement for consistency analysis
        agent_pos = environment.agents_positions[self.agent_id]
        self.movement_history.append(agent_pos.copy())
        
        return enhanced_state
    
    def decompose_hierarchical_goals(self, current_pos: np.ndarray, target_pos: np.ndarray, 
                                   environment: MultiAgentEnvironment) -> List[np.ndarray]:
        """
        Hierarchical goal decomposition with constraint awareness
        """
        total_distance = np.linalg.norm(target_pos - current_pos)
        
        # Adaptive subgoal spacing based on environment complexity
        obstacle_density = len(environment.obstacles) / (environment.grid_size ** 2)
        base_spacing = 2.0
        adaptive_spacing = base_spacing * (1 + obstacle_density)
        
        num_subgoals = max(1, int(total_distance / adaptive_spacing))
        num_subgoals = min(num_subgoals, 5)  # Cap at 5 subgoals
        
        subgoals = []
        for i in range(1, num_subgoals + 1):
            ratio = i / num_subgoals
            subgoal_pos = current_pos + ratio * (target_pos - current_pos)
            
            # Adjust subgoal to avoid obstacles
            subgoal_pos = self._find_safe_subgoal_position(subgoal_pos, environment)
            subgoals.append(subgoal_pos)
        
        return subgoals
    
    def _find_safe_subgoal_position(self, original_pos: np.ndarray, 
                                  environment: MultiAgentEnvironment) -> np.ndarray:
        """Find a safe position for subgoal, avoiding obstacles"""
        # Check if original position is safe
        if not environment._check_collision(original_pos):
            return original_pos
        
        # Search for alternative positions in a spiral pattern
        search_radius = 1.0
        for radius in np.linspace(0.5, search_radius, 10):
            for angle in np.linspace(0, 2*np.pi, 8):
                candidate_pos = original_pos + radius * np.array([np.cos(angle), np.sin(angle)])
                
                # Check bounds
                if (0 <= candidate_pos[0] <= environment.grid_size and 
                    0 <= candidate_pos[1] <= environment.grid_size):
                    if not environment._check_collision(candidate_pos):
                        return candidate_pos
        
        # If no safe position found, return original (will need to handle collision)
        return original_pos
    
    def update_subgoal_progress(self, current_pos: np.ndarray):
        """Update subgoal progress and advance to next subgoal if reached"""
        if not self.subgoals or self.current_subgoal_index >= len(self.subgoals):
            return False
        
        current_subgoal = self.subgoals[self.current_subgoal_index]
        distance_to_subgoal = np.linalg.norm(current_pos - current_subgoal)
        
        if distance_to_subgoal < self.subgoal_threshold:
            self.current_subgoal_index += 1
            return True
        
        return False
    
    def select_action(self, state: np.ndarray, environment: MultiAgentEnvironment, 
                     exploration_noise: float = 0.1) -> np.ndarray:
        """
        Enhanced action selection with constraint awareness and hierarchical goals
        """
        # Compute relational state
        relational_state = self.compute_relational_state(state, environment)
        
        # Update hierarchical goals
        current_pos = environment.agents_positions[self.agent_id]
        target_pos = environment.targets[self.agent_id]
        
        # Decompose goals if needed
        if not self.subgoals:
            self.subgoals = self.decompose_hierarchical_goals(current_pos, target_pos, environment)
            self.current_subgoal_index = 0
        
        # Update subgoal progress
        self.update_subgoal_progress(current_pos)
        
        # Determine current target (subgoal or final target)
        if self.current_subgoal_index < len(self.subgoals):
            current_target = self.subgoals[self.current_subgoal_index]
        else:
            current_target = target_pos
        
        # Get action from actor network
        state_tensor = torch.FloatTensor(relational_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # Add exploration noise
        if exploration_noise > 0:
            noise = np.random.normal(0, exploration_noise, size=action.shape)
            action = action + noise
        
        # Constraint-aware action modification
        action = self._apply_constraint_awareness(action, relational_state, environment)
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def _apply_constraint_awareness(self, action: np.ndarray, state: np.ndarray, 
                                  environment: MultiAgentEnvironment) -> np.ndarray:
        """Apply constraint awareness to modify actions"""
        current_pos = environment.agents_positions[self.agent_id]
        
        # Extract constraint features from state
        constraint_pressure = state[10]  # Index 10 is constraint pressure
        trap_pressure = state[12]  # Index 12 is trap pressure
        
        # If constraint pressure is high, modify action to move away from constraints
        if constraint_pressure > self.constraint_threshold:
            # Find direction away from nearest obstacle
            escape_direction = self._compute_escape_direction(current_pos, environment)
            
            # Blend original action with escape direction
            constraint_weight = min(1.0, constraint_pressure * 2)  # Scale up response
            action = (1 - constraint_weight) * action + constraint_weight * escape_direction
        
        # If trapped, use more aggressive escape behavior
        if trap_pressure > 0.5:
            escape_direction = self._compute_escape_direction(current_pos, environment)
            action = 0.3 * action + 0.7 * escape_direction  # Prioritize escape
        
        return action
    
    def _compute_escape_direction(self, current_pos: np.ndarray, 
                                environment: MultiAgentEnvironment) -> np.ndarray:
        """Compute direction to escape constraints"""
        escape_direction = np.array([0.0, 0.0])
        
        # Compute repulsion from obstacles
        for obstacle in environment.obstacles:
            direction_from_obstacle = current_pos - obstacle['center']
            distance = np.linalg.norm(direction_from_obstacle)
            
            if distance < obstacle['radius'] + 2.0:  # Within influence range
                # Normalize direction and weight by inverse distance
                if distance > 1e-6:
                    direction_from_obstacle = direction_from_obstacle / distance
                    weight = (obstacle['radius'] + 2.0 - distance) / 2.0
                    escape_direction += weight * direction_from_obstacle
        
        # Normalize final escape direction
        if np.linalg.norm(escape_direction) > 1e-6:
            escape_direction = escape_direction / np.linalg.norm(escape_direction)
        else:
            # Random direction if no clear escape
            angle = np.random.uniform(0, 2*np.pi)
            escape_direction = np.array([np.cos(angle), np.sin(angle)])
        
        return escape_direction
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def train(self) -> Dict[str, float]:
        """Train the agent using DDPG algorithm with relational state features"""
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Train Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions)
            target_q = rewards + (self.gamma * next_q_values * ~dones).squeeze()
        
        current_q = self.critic(states, actions).squeeze()
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Train Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update target networks
        self._soft_update(self.critic_target, self.critic)
        self._soft_update(self.actor_target, self.actor)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def _soft_update(self, target_network: nn.Module, source_network: nn.Module):
        """Soft update target network parameters"""
        for target_param, source_param in zip(target_network.parameters(), 
                                            source_network.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def reset_episode(self):
        """Reset agent state for new episode"""
        self.subgoals = []
        self.current_subgoal_index = 0
        self.previous_state = None
        self.movement_history.clear()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for analysis"""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        
        metrics = {
            'average_reward': np.mean(self.episode_rewards),
            'recent_average_reward': np.mean(recent_rewards),
            'reward_std': np.std(self.episode_rewards),
            'total_episodes': len(self.episode_rewards)
        }
        
        if self.cooperation_history:
            metrics['average_cooperation'] = np.mean(self.cooperation_history)
        if self.competition_history:
            metrics['average_competition'] = np.mean(self.competition_history)
        
        return metrics

def run_training_simulation(num_episodes: int = 20, 
                          num_agents: int = 2, 
                          grid_size: int = 10,
                          max_steps_per_episode: int = 1000):
    """
    Main training simulation with integrated relational state features
    """
    # Initialize environment
    env = MultiAgentEnvironment(num_agents=num_agents, grid_size=grid_size)
    env.max_steps = max_steps_per_episode
    
    # Initialize agents
    state_dim = 32  # Fixed relational state dimension
    action_dim = 2  # 2D movement
    
    agents = []
    for i in range(num_agents):
        agent = GameTheoryAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            agent_id=i,
            lr_actor=1e-4,
            lr_critic=1e-3,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        agents.append(agent)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    cooperation_scores = []
    competition_scores = []
    constraint_violations = []
    
    # Training loop
    logger.info(f"Starting training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset environment and agents
        states = env.reset()
        for agent in agents:
            agent.reset_episode()
        
        episode_reward = [0.0] * num_agents
        episode_cooperation = []
        episode_competition = []
        episode_constraints = []
        
        step_count = 0
        done = False
        
        while not done and step_count < max_steps_per_episode:
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(agents):
                # Compute exploration noise (decay over episodes)
                exploration_noise = max(0.01, 0.3 * (1 - episode / num_episodes))
                action = agent.select_action(states[i], env, exploration_noise)
                actions.append(action)
            
            # Execute actions in environment
            next_states, rewards, done, info = env.step(actions)
            
            # Store experiences and train agents
            for i, agent in enumerate(agents):
                agent.store_experience(states[i], actions[i], rewards[i], next_states[i], done)
                
                # Train if enough experiences
                if len(agent.memory) >= agent.batch_size and step_count % 4 == 0:
                    training_info = agent.train()
                
                episode_reward[i] += rewards[i]
            
            # Track cooperation and competition metrics
            coop_score = np.mean([state[19] for state in next_states])  # Cooperation potential
            comp_score = np.mean([state[20] for state in next_states])  # Competition pressure
            constraint_score = np.mean([state[10] for state in next_states])  # Constraint pressure
            
            episode_cooperation.append(coop_score)
            episode_competition.append(comp_score)
            episode_constraints.append(constraint_score)
            
            states = next_states
            step_count += 1
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        cooperation_scores.append(np.mean(episode_cooperation) if episode_cooperation else 0.0)
        competition_scores.append(np.mean(episode_competition) if episode_competition else 0.0)
        constraint_violations.append(np.mean(episode_constraints) if episode_constraints else 0.0)
        
        # Update agent episode rewards
        for i, agent in enumerate(agents):
            agent.episode_rewards.append(episode_reward[i])
            if episode_cooperation:
                agent.cooperation_history.append(np.mean(episode_cooperation))
            if episode_competition:
                agent.competition_history.append(np.mean(episode_competition))
        
        # Log progress
        if (episode + 1) % 5 == 0 or episode == 0:
            avg_reward = np.mean([np.mean(rewards) for rewards in episode_rewards[-5:]])
            avg_length = np.mean(episode_lengths[-5:])
            avg_cooperation = np.mean(cooperation_scores[-5:])
            avg_competition = np.mean(competition_scores[-5:])
            avg_constraints = np.mean(constraint_violations[-5:])
            
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            logger.info(f"  Average Reward: {avg_reward:.3f}")
            logger.info(f"  Average Length: {avg_length:.1f}")
            logger.info(f"  Cooperation Score: {avg_cooperation:.3f}")
            logger.info(f"  Competition Score: {avg_competition:.3f}")
            logger.info(f"  Constraint Violations: {avg_constraints:.3f}")
            logger.info("-" * 50)
    
    # Final analysis and results
    logger.info("Training completed!")
    logger.info("=" * 60)
    logger.info("FINAL RESULTS:")
    logger.info("=" * 60)
    
    # Overall performance
    total_avg_reward = np.mean([np.mean(rewards) for rewards in episode_rewards])
    final_avg_reward = np.mean([np.mean(rewards) for rewards in episode_rewards[-5:]])
    reward_improvement = final_avg_reward - np.mean([np.mean(rewards) for rewards in episode_rewards[:5]])
    
    logger.info(f"Overall Average Reward: {total_avg_reward:.3f}")
    logger.info(f"Final 5 Episodes Average Reward: {final_avg_reward:.3f}")
    logger.info(f"Reward Improvement: {reward_improvement:+.3f}")
    
    # Cooperation and competition analysis
    total_cooperation = np.mean(cooperation_scores)
    total_competition = np.mean(competition_scores)
    total_constraints = np.mean(constraint_violations)
    
    logger.info(f"Average Cooperation Score: {total_cooperation:.3f}")
    logger.info(f"Average Competition Score: {total_competition:.3f}")
    logger.info(f"Average Constraint Violations: {total_constraints:.3f}")
    
    # Agent-specific performance
    logger.info("\nAgent-specific Performance:")
    for i, agent in enumerate(agents):
        metrics = agent.get_performance_metrics()
        logger.info(f"Agent {i}:")
        logger.info(f"  Average Reward: {metrics.get('average_reward', 0):.3f}")
        logger.info(f"  Recent Average: {metrics.get('recent_average_reward', 0):.3f}")
        logger.info(f"  Cooperation: {metrics.get('average_cooperation', 0):.3f}")
        logger.info(f"  Competition: {metrics.get('average_competition', 0):.3f}")
    
    # Relational state feature analysis
    logger.info("\nRelational State Features Analysis:")
    logger.info("✅ Progress and remaining ratios implemented")
    logger.info("✅ Multi-scale gap analysis (linear, log, sqrt)")
    logger.info("✅ Constraint features and trap pressure")
    logger.info("✅ Adaptive phase identification")
    logger.info("✅ Efficiency metrics with proper bounds")
    logger.info("✅ Game theory features (cooperation/competition)")
    logger.info("✅ Hierarchical goal decomposition")
    logger.info("✅ Constraint-aware action selection")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'cooperation_scores': cooperation_scores,
        'competition_scores': competition_scores,
        'constraint_violations': constraint_violations,
        'agents': agents,
        'environment': env,
        'final_metrics': {
            'total_avg_reward': total_avg_reward,
            'final_avg_reward': final_avg_reward,
            'reward_improvement': reward_improvement,
            'total_cooperation': total_cooperation,
            'total_competition': total_competition,
            'total_constraints': total_constraints
        }
    }

# Example usage and testing
if __name__ == "__main__":
    # Run the training simulation
    results = run_training_simulation(
        num_episodes=20,
        num_agents=2,
        grid_size=10,
        max_steps_per_episode=1000
    )
    
    # Additional analysis can be performed here
    print("\nTraining completed successfully!")
    print("Results available in 'results' dictionary")
    
    # Example of accessing results
    print(f"\nFinal performance metrics:")
    for key, value in results['final_metrics'].items():
        print(f"{key}: {value:.4f}")