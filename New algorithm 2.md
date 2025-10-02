# Technical Implementation Guide: Hierarchical Relational Reinforcement Learning

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Core Components](#core-components)
3. [Implementation Details](#implementation-details)
4. [Training Procedures](#training-procedures)
5. [Evaluation Methods](#evaluation-methods)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Extension Guidelines](#extension-guidelines)

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Environment Interface                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ Goal Decomposer │    │ Constraint      │                 │
│  │                 │    │ Analyzer        │                 │
│  └─────────────────┘    └─────────────────┘                 │
├─────────────────────────────────────────────────────────────┤
│                Hierarchical State Encoder                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ Phase-Adaptive  │    │ Experience      │                 │
│  │ Q-Network       │    │ Replay Buffer   │                 │
│  └─────────────────┘    └─────────────────┘                 │
├─────────────────────────────────────────────────────────────┤
│                    Action Selection Layer                    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input Processing**: Environment state → Relational feature extraction
2. **Goal Management**: Target decomposition → Subgoal tracking
3. **Network Processing**: Features → Phase classification → Q-value computation
4. **Action Selection**: Q-values + constraints → Valid action selection
5. **Learning**: Experience storage → Batch sampling → Network updates

## Core Components

### 1. HierarchicalState Class

**Purpose**: Convert raw environment states into scale-invariant relational features.

**Key Features**:
- Multi-scale gap analysis
- Constraint proximity computation
- Phase identification
- Efficiency metrics

**Critical Implementation Details**:

```python
class HierarchicalState:
    def __init__(self, current, target, step, max_steps, forbidden_states=None):
        self.current = current
        self.target = target
        self.step = step
        self.max_steps = max_steps
        self.forbidden_states = forbidden_states or set()
    
    def to_features(self):
        """
        Returns 12-dimensional feature vector:
        [0] progress_ratio: current/target (0-1+)
        [1] remaining_ratio: (target-current)/target (0-1)
        [2] time_ratio: step/max_steps (0-1)
        [3] log_gap: log-scaled distance to target
        [4] gap_magnitude: relative gap size
        [5] is_close: binary flag for proximity
        [6] is_far: binary flag for distance
        [7] danger_proximity: distance to nearest forbidden state
        [8] constraint_pressure: path blockage measure
        [9] phase: problem-solving phase (0/1/2)
        [10] theoretical_min_steps: optimal step count
        [11] efficiency_ratio: time pressure measure
        """
        if self.target == 0:
            return np.zeros(12)
        
        # Core relational features
        progress_ratio = self.current / self.target
        remaining_ratio = (self.target - self.current) / self.target
        time_ratio = self.step / self.max_steps
        
        # Multi-scale analysis
        gap = abs(self.target - self.current)
        log_gap = math.log(gap + 1) / math.log(self.target + 1)
        gap_magnitude = gap / self.target
        
        # Strategic indicators
        is_close = 1.0 if gap <= 10 else 0.0
        is_far = 1.0 if gap >= self.target * 0.5 else 0.0
        
        # Constraint analysis
        danger_proximity = self._compute_danger_proximity()
        constraint_pressure = self._compute_constraint_pressure()
        
        # Phase and efficiency
        phase = self._identify_phase()
        theoretical_min_steps = self._compute_min_steps()
        efficiency_ratio = theoretical_min_steps / (self.max_steps - self.step + 1)
        
        return np.array([
            progress_ratio, remaining_ratio, time_ratio,
            log_gap, gap_magnitude, is_close, is_far,
            danger_proximity, constraint_pressure,
            phase, theoretical_min_steps, efficiency_ratio
        ])
```

**Performance Considerations**:
- Feature computation is O(|forbidden_states|) 
- Cache expensive computations when possible
- Normalize all features to [0,1] range for neural network stability

### 2. HierarchicalQNetwork Architecture

**Purpose**: Multi-head neural network with phase-specific specialization.

**Architecture Specifications**:

```python
class HierarchicalQNetwork(nn.Module):
    def __init__(self, state_dim=12, action_dim=3, hidden_dim=256):
        super().__init__()
        
        # Shared feature extractor (critical for transfer learning)
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Prevent overfitting on small datasets
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Phase-specific heads (enables specialized strategies)
        self.exploration_head = self._build_head(hidden_dim, action_dim)
        self.navigation_head = self._build_head(hidden_dim, action_dim)
        self.precision_head = self._build_head(hidden_dim, action_dim)
        
        # Phase classifier (learns to identify problem phases)
        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )
    
    def _build_head(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim)
        )
    
    def forward(self, state_features):
        # Extract shared representations
        features = self.feature_extractor(state_features)
        
        # Classify current phase
        phase_probs = self.phase_classifier(features)
        
        # Compute phase-specific Q-values
        exploration_q = self.exploration_head(features)
        navigation_q = self.navigation_head(features)
        precision_q = self.precision_head(features)
        
        # Weighted combination based on phase probabilities
        q_values = (phase_probs[:, 0:1] * exploration_q + 
                   phase_probs[:, 1:2] * navigation_q + 
                   phase_probs[:, 2:3] * precision_q)
        
        return q_values, phase_probs
```

**Design Rationale**:
- **Shared Feature Extractor**: Enables transfer learning across phases
- **Phase-Specific Heads**: Allows specialized strategies for different problem stages
- **Soft Attention**: Phase probabilities enable smooth transitions between strategies
- **Dropout**: Prevents overfitting on small training datasets

### 3. GoalDecompositionAgent

**Purpose**: Main agent class coordinating all components.

**Key Responsibilities**:
- Goal decomposition and subgoal management
- Experience collection and replay
- Network training and target updates
- Action selection with constraint handling

**Critical Methods**:

```python
class GoalDecompositionAgent:
    def decompose_target(self, current, target):
        """
        Automatically decompose large targets into manageable subgoals.
        
        Strategy:
        - Targets ≤ 50: Direct approach
        - Targets > 50: Create intermediate waypoints
        - Waypoint spacing: ~75 units (empirically optimal)
        """
        gap = abs(target - current)
        
        if gap <= 50:
            return [target]
        
        num_subgoals = max(2, gap // 75)
        step_size = gap // num_subgoals
        direction = 1 if target > current else -1
        
        subgoals = []
        for i in range(1, num_subgoals):
            subgoal = current + (step_size * i * direction)
            subgoals.append(subgoal)
        
        subgoals.append(target)
        return subgoals
    
    def choose_action(self, current, target, step, max_steps, forbidden_states=None, training=True):
        """
        Action selection with hierarchical reasoning and constraint handling.
        """
        # Update subgoals if needed
        if not self.subgoal_stack:
            self.subgoal_stack = self.decompose_target(current, target)
            self.current_subgoal = self.subgoal_stack.pop(0)
        
        # Check subgoal completion
        if current == self.current_subgoal and self.subgoal_stack:
            self.current_subgoal = self.subgoal_stack.pop(0)
        
        # Use current subgoal for decision making
        working_target = self.current_subgoal if self.current_subgoal else target
        
        # Adaptive exploration based on problem difficulty
        difficulty = abs(target - current)
        epsilon = self.epsilon_schedule(self.step_count, difficulty)
        
        if training and random.random() < epsilon:
            return random.choice(self.actions)
        
        # Neural network decision
        state_features = self.get_state_features(current, working_target, step, max_steps, forbidden_states)
        state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
        
        with torch.no_grad():
            q_values, phase_probs = self.q_network(state_tensor)
        
        # Constraint-aware action selection
        valid_actions = []
        for i, action in enumerate(self.actions):
            next_state = current + action
            if forbidden_states and next_state in forbidden_states:
                continue
            valid_actions.append((i, action))
        
        if not valid_actions:
            # Emergency backtracking
            return -self.actions[0] if -self.actions[0] in self.actions else self.actions[0]
        
        # Select highest-