
"""
self_improving_training.py

Complete, runnable Python implementation of a Self-Improving Training Loop.
- 2-layer neural network (tanh hidden)
- Exact backprop implemented (NumPy)
- MetaOptimizer supporting: GD, Adam, Adaptive LR, Evolutionary
- SelfImprovingTrainingLoop that ties everything and runs training
- Demo section at the end that runs a short training session and prints a summary

Usage:
    python self_improving_training.py
"""
import numpy as np
import copy
import time
from enum import Enum
from dataclasses import dataclass

np.random.seed(1)

# -------------------------
# Utilities and enums
# -------------------------
class OptimizationStrategy(Enum):
    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    ADAPTIVE_LR = "adaptive_lr"
    EVOLUTIONARY = "evolutionary"

def mse(pred, target):
    pred = np.array(pred)
    target = np.array(target)
    return float(np.mean((pred - target) ** 2))

def create_synthetic_batch(n_samples, input_dim):
    X = np.random.uniform(-1, 1, size=(n_samples, input_dim))
    weights = np.array([0.6, -0.4, 0.2] + [0.0] * max(0, input_dim-3))
    y = (X * weights[:input_dim]).sum(axis=1, keepdims=True) + 0.1 * (np.random.rand(n_samples,1) - 0.5)
    return X, y

# -------------------------
# Hyperparameters & Metrics
# -------------------------
@dataclass
class TrainingMetrics:
    iteration: int
    loss: float
    accuracy: float
    learning_rate: float
    strategy_used: str
    computation_time_ms: float
    gradient_norm: float

class HyperParameters:
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=1e-4, min_improvement=1e-6):
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.min_improvement = float(min_improvement)
    def mutate(self, mutation_rate=0.2):
        new = copy.deepcopy(self)
        if np.random.rand() < mutation_rate:
            new.learning_rate *= (0.5 + np.random.rand() * 1.5)
            new.learning_rate = float(np.clip(new.learning_rate, 1e-6, 1.0))
        if np.random.rand() < mutation_rate:
            new.momentum = float(np.clip(new.momentum + (np.random.rand()-0.5)*0.2, 0.0, 0.999))
        if np.random.rand() < mutation_rate:
            new.weight_decay *= (0.1 + np.random.rand()*9.9)
            new.weight_decay = float(np.clip(new.weight_decay, 1e-8, 1e-2))
        return new

# -------------------------
# Simple Neural Network (2-layer) with backprop
# -------------------------
class SimpleNeuralNetwork:
    def __init__(self, input_dim=5, hidden_dim=20, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Xavier init
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / max(1,input_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / max(1,hidden_dim))
        self.b2 = np.zeros((1, output_dim))
        self.cache = {}
    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (batch, features)")
        self.cache['X'] = X
        self.cache['z1'] = X @ self.W1 + self.b1  # (B,H)
        self.cache['a1'] = np.tanh(self.cache['z1'])  # (B,H)
        self.cache['z2'] = self.cache['a1'] @ self.W2 + self.b2  # (B,O)
        return self.cache['z2']
    def loss(self, preds, targets):
        return mse(preds, targets)
    def backward(self, preds, targets):
        B = preds.shape[0]
        dloss_dz2 = 2.0 * (preds - targets) / B  # (B,O)
        dW2 = self.cache['a1'].T @ dloss_dz2  # (H,O)
        db2 = dloss_dz2.sum(axis=0, keepdims=True)  # (1,O)
        da1 = dloss_dz2 @ self.W2.T  # (B,H)
        dz1 = da1 * (1 - np.tanh(self.cache['z1'])**2)  # (B,H)
        dW1 = self.cache['X'].T @ dz1  # (D,H)
        db1 = dz1.sum(axis=0, keepdims=True)  # (1,H)
        grads = {'W1': dW1, 'b1': db1.reshape(-1), 'W2': dW2, 'b2': db2.reshape(-1)}
        return grads
    def get_parameters(self):
        return {'W1': self.W1.copy(), 'b1': self.b1.copy().reshape(-1),
                'W2': self.W2.copy(), 'b2': self.b2.copy().reshape(-1)}
    def set_parameters(self, params):
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy().reshape(1,-1)
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy().reshape(1,-1)

# -------------------------
# MetaOptimizer
# -------------------------
class MetaOptimizer:
    def __init__(self):
        self.strategy_performance = {s: {'uses':0,'success':0,'avg_impr':0.0,'recent':[]} for s in OptimizationStrategy}
        self.current_strategy = OptimizationStrategy.GRADIENT_DESCENT
        self.adam_state = {}
        self.t = 0
    def select_strategy(self, metrics_history):
        if len(metrics_history) < 3:
            return self.current_strategy
        recent = metrics_history[-3:]
        improv = recent[0]['loss'] - recent[-1]['loss']
        if improv < 1e-6 and np.random.rand() < 0.5:
            choices = [s for s in OptimizationStrategy if s != self.current_strategy]
            self.current_strategy = np.random.choice(choices)
        return self.current_strategy
    def update_parameters(self, model, grads, hyperparams, strategy):
        params = model.get_parameters()
        lr = hyperparams.learning_rate
        if strategy == OptimizationStrategy.GRADIENT_DESCENT:
            params['W1'] -= lr * grads['W1']
            params['b1'] -= lr * grads['b1']
            params['W2'] -= lr * grads['W2']
            params['b2'] -= lr * grads['b2']
        elif strategy == OptimizationStrategy.ADAPTIVE_LR:
            for k in ['W1','W2','b1','b2']:
                g = grads[k]
                params[k] -= (lr/(1.0 + 0.01 * np.sqrt(np.mean(g**2)))) * g
        elif strategy == OptimizationStrategy.EVOLUTIONARY:
            for k in ['W1','W2','b1','b2']:
                perturb = 0.001 * (np.random.randn(*params[k].shape))
                params[k] -= lr * grads[k] + perturb
        elif strategy == OptimizationStrategy.ADAM:
            self.t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for k in ['W1','W2','b1','b2']:
                g = grads[k]
                if k not in self.adam_state:
                    self.adam_state[k] = {'m':np.zeros_like(g),'v':np.zeros_like(g)}
                st = self.adam_state[k]
                st['m'] = beta1 * st['m'] + (1-beta1) * g
                st['v'] = beta2 * st['v'] + (1-beta2) * (g**2)
                mhat = st['m'] / (1 - beta1**self.t)
                vhat = st['v'] / (1 - beta2**self.t)
                params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        else:
            raise ValueError("Unknown strategy")
        model.set_parameters(params)

# -------------------------
# SelfImprovingTrainingLoop
# -------------------------
class SelfImprovingTrainingLoop:
    def __init__(self, input_dim=5, hidden_dim=20, output_dim=1, seed=0):
        np.random.seed(seed)
        self.model = SimpleNeuralNetwork(input_dim, hidden_dim, output_dim)
        self.hyper = HyperParameters(learning_rate=0.01)
        self.meta = MetaOptimizer()
        self.metrics_history = []
        self.best_loss = float('inf')
        self.best_params = None
        self.iteration = 0
    def train_step(self, X, y):
        t0 = time.time()
        preds = self.model.forward(X)
        loss = self.model.loss(preds, y)
        grads = self.model.backward(preds, y)
        grad_norm = np.sqrt(sum([np.sum(g**2) for g in grads.values()]))
        strat = self.meta.select_strategy(self.metrics_history)
        self.meta.update_parameters(self.model, grads, self.hyper, strat)
        mae = float(np.mean(np.abs(preds - y)))
        acc = 1.0/(1.0 + mae)
        t_ms = (time.time() - t0) * 1000.0
        metrics = {'iteration': self.iteration, 'loss': float(loss), 'accuracy': float(acc),
                   'learning_rate': self.hyper.learning_rate, 'strategy_used': strat.value,
                   'computation_time_ms': t_ms, 'gradient_norm': float(grad_norm)}
        self.metrics_history.append(metrics)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = self.model.get_parameters()
        self.iteration += 1
        if self.iteration % 10 == 0 and len(self.metrics_history) >= 10:
            recent = [m['loss'] for m in self.metrics_history[-10:]]
            if recent[0] - recent[-1] < self.hyper.min_improvement:
                old = self.hyper.learning_rate
                if np.random.rand() < 0.5:
                    self.hyper.learning_rate *= (0.5 + np.random.rand()*0.5)
                else:
                    self.hyper.learning_rate *= (1 + np.random.rand()*0.5)
                self.hyper.learning_rate = float(np.clip(self.hyper.learning_rate, 1e-6, 1.0))
        return metrics
    def train(self, n_iters=50, batch_size=32, verbose=True):
        for i in range(n_iters):
            X,y = create_synthetic_batch(batch_size, self.model.input_dim)
            m = self.train_step(X,y)
            if verbose and (i==0 or (i+1)%10==0 or i==n_iters-1):
                print(f"Iter {i+1}/{n_iters} - loss {m['loss']:.6f} acc {m['accuracy']:.4f} strat {m['strategy_used']} lr {m['learning_rate']:.6f}")
        return self.metrics_history
    def summary(self):
        if not self.metrics_history:
            return {}
        initial = self.metrics_history[0]['loss']
        final = self.metrics_history[-1]['loss']
        usage = {}
        for s in OptimizationStrategy:
            usage[s.value] = sum(1 for m in self.metrics_history if m['strategy_used']==s.value)
        return {'total_iters': len(self.metrics_history), 'initial_loss': initial, 'final_loss': final,
                'best_loss': float(self.best_loss), 'strategy_usage': usage, 'final_lr': self.hyper.learning_rate}

# -------------------------
# Demo when run as script
# -------------------------
if __name__ == "__main__":
    loop = SelfImprovingTrainingLoop(input_dim=5, hidden_dim=12, output_dim=1, seed=2)
    print("Starting demo training (40 iterations, batch=64)...")
    history = loop.train(n_iters=40, batch_size=64, verbose=True)
    print("\nDemo finished. Summary:")
    summary = loop.summary()
    print(f" - total iters: {summary['total_iters']}")
    print(f" - initial loss: {summary['initial_loss']:.6f}")
    print(f" - final loss: {summary['final_loss']:.6f}")
    print(f" - best loss: {summary['best_loss']:.6f}")
    print(" - strategy usage:")
    for k,v in summary['strategy_usage'].items():
        print(f"    {k}: {v} iterations")
    Xv, yv = create_synthetic_batch(200, loop.model.input_dim)
    preds_v = loop.model.forward(Xv)
    val_loss = mse(preds_v, yv)
    print(f"\nValidation loss on fresh data (200 samples): {val_loss:.6f}")
