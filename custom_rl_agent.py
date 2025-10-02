import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

# Define a simple activation function that can be modulated by a parameter 'p'
def parametric_relu(x, p):
    """Parametric ReLU activation: max(0, x) + p * min(0, x)"""
    return torch.max(torch.zeros_like(x), x) + p * torch.min(torch.zeros_like(x), x)

# Custom module for parametric activation
class ParametricReLU(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return parametric_relu(x, self.p)

class CustomRLAgent:
    def __init__(self, observation_space, action_space, learning_rate=0.01, gamma=0.99):
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma

        obs_dim = observation_space.shape[0]
        action_dim = action_space.n if isinstance(action_space, gym.spaces.Discrete) else action_space.shape[0]

        self.policy_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_logits = self.policy_network(state_tensor)
        action_distribution = distributions.Categorical(logits=action_logits)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item(), log_prob

    def update(self):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        # Normalize rewards (optional, but often helps)
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        torch.stack(policy_loss).sum().backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

    def learn(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        if done:
            self.update()


class ActorCriticAgent(CustomRLAgent):
    def __init__(self, observation_space, action_space, learning_rate=0.01, gamma=0.99):
        super().__init__(observation_space, action_space, learning_rate, gamma)

        obs_dim = observation_space.shape[0]

        self.value_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Optimizer for both actor and critic networks
        self.optimizer = optim.Adam([
            {'params': self.policy_network.parameters(), 'lr': learning_rate},
            {'params': self.value_network.parameters(), 'lr': learning_rate}
        ])

        self.log_probs = []
        self.rewards = []
        self.values = [] # To store value estimates


    def learn(self, state, action, reward, next_state, done):
        # This method will now collect rewards and value estimates
        self.rewards.append(reward)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        value = self.value_network(state_tensor)
        self.values.append(value)

        if done:
            self.update()

    def update(self):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        values = torch.stack(self.values).squeeze()

        # Calculate advantage
        advantage = discounted_rewards - values

        # Calculate actor loss
        actor_loss = []
        for log_prob, adv in zip(self.log_probs, advantage.detach()): # Use .detach() for advantage
            actor_loss.append(-log_prob * adv)
        actor_loss = torch.stack(actor_loss).sum()

        # Calculate critic loss (MSE)
        critic_loss = F.mse_loss(values, discounted_rewards)

        # Total loss
        loss = actor_loss + critic_loss # You might add a weighting factor for critic_loss

        # Perform optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffers
        self.log_probs = []
        self.rewards = []
        self.values = []

class PBWCActorCriticAgent(ActorCriticAgent):
    def __init__(self, observation_space, action_space, learning_rate=0.01, gamma=0.99):
        super().__init__(observation_space, action_space, learning_rate, gamma)

        obs_dim = observation_space.shape[0]
        action_dim = action_space.n if isinstance(action_space, gym.spaces.Discrete) else action_space.shape[0]

        # Initialize trainable PBWC parameters 'p' for each layer.
        self.policy_p1 = nn.Parameter(torch.tensor([1.0], dtype=torch.float32)) # Example: initial p for policy hidden layer
        self.value_p1 = nn.Parameter(torch.tensor([1.0], dtype=torch.float32)) # Example: initial p for value hidden layer


        # Redefine policy network to use custom parametric activation module
        self.policy_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            ParametricReLU(self.policy_p1), # Use custom module
            nn.Linear(64, action_dim)
        )

        # Redefine value network to use custom parametric activation module
        self.value_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            ParametricReLU(self.value_p1), # Use custom module
            nn.Linear(64, 1)
        )

        # Get parameters excluding the 'p' parameters which are already in the networks
        policy_params_without_p = [param for name, param in self.policy_network.named_parameters() if 'p' not in name]
        value_params_without_p = [param for name, param in self.value_network.named_parameters() if 'p' not in name]

        # Update the optimizer to include the network parameters and the explicit 'p' parameters
        self.optimizer = optim.Adam([
            {'params': policy_params_without_p, 'lr': learning_rate},
            {'params': value_params_without_p, 'lr': learning_rate},
            {'params': [self.policy_p1, self.value_p1], 'lr': learning_rate} # Explicitly add 'p' parameters
        ])


        # These buffers are inherited from the base class but are listed for clarity
        self.log_probs = []
        self.rewards = []
        self.values = []

    def update(self):
        # Implement Bias-Integrated Integral (BII) concept for discounted rewards
        # Incorporate a learned bias term (e.g., self.value_p1 could influence this)
        # For simplicity, let's add a small bias related to the learned parameter
        bias_term = self.value_p1.item() * 0.1 # Example: bias proportional to value_p1

        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.rewards):
            # Apply bias during reward accumulation
            cumulative_reward = reward + self.gamma * cumulative_reward + bias_term
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        values = torch.stack(self.values).squeeze()

        # Consider Weight-Series Convergence (WSC) - dynamically adjust gamma?
        # For this example, let's keep gamma fixed, but acknowledge this as a PBWC point.
        # A more complex implementation might adjust gamma based on reward variance or return instability.

        # Calculate advantage (potentially influenced by PBWC concepts)
        # The advantage calculation itself is standard, but the values and discounted_rewards
        # are now influenced by the PBWC bias in BII.
        advantage = discounted_rewards - values

        # Calculate actor loss (potentially influenced by PBWC concepts like PD)
        # PD suggests modifying gradient calculation. Here, we'll use the standard policy gradient
        # loss with the PBWC-influenced advantage. A more advanced PBWC might modify the log_prob
        # calculation or the loss function structure itself.
        actor_loss = []
        for log_prob, adv in zip(self.log_probs, advantage.detach()): # Use .detach() for advantage in actor loss
            actor_loss.append(-log_prob * adv)
        actor_loss = torch.stack(actor_loss).sum()

        # Calculate critic loss (MSE, potentially influenced by PBWC concepts)
        # BII influenced the target (discounted_rewards). Critic loss is standard MSE on PBWC-influenced targets.
        # A PBWC "mathematical loss" might be a different function or include other terms.
        critic_loss = F.mse_loss(values, discounted_rewards)

        # Total loss (potentially influenced by PBWC concepts)
        # A simple sum here. PBWC might suggest a weighted sum or a more complex combination based on 'p'.
        loss = actor_loss + critic_loss # You might add a weighting factor for critic_loss

        # Perform optimization (incorporates PBWC Evolution Shift/Update Rule)
        # The optimizer applies the gradient updates to all parameters (w, b, and p)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffers
        self.log_probs = []
        self.rewards = []
        self.values = []
