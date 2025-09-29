"""
Manus DL - Complete Improved Engine

This is a single-file experimental deep-learning/autodiff engine implemented
for clarity and hands-on tinkering in Colab or local Python.

Features:
- Node-based autodiff with topological backward
- float64 gradient accumulation toggle
- Parameter and Module abstractions
- Layers: Linear, ReLU, Softmax
- Sequential container
- Losses: MSELoss and SoftmaxCrossEntropyLoss
- Optimizers: SGD (with momentum/weight decay) and Adam
- Gradient clipping utility
- Simple NumpyDataset and DataLoader
- Training helpers and save/load utilities
- Small demo in __main__

This file is intended to be a practical, self-contained baseline you can
copy into Google Colab and experiment with. It's written for readability
(rather than raw performance).
"""

from __future__ import annotations
import numpy as np
import math
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ------------------------- Config ---------------------------------------
# Toggle: use float64 internal accumulation for stability
ENABLE_HIGH_PRECISION_ACCUMULATION: bool = True

# ------------------------- Utilities ------------------------------------

def set_seed(seed: int):
    np.random.seed(seed)


def _sum_to_shape(arr: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce `arr` to `shape` by summation across broadcasted axes.
    This handles leading/trailing broadcast dims conservatively.
    """
    arr = np.asarray(arr)
    if arr.shape == shape:
        return arr
    # reduce leading dims
    while arr.ndim > len(shape):
        arr = arr.sum(axis=0)
    # reduce axes where target has size 1
    for i, (a_dim, s_dim) in enumerate(zip(arr.shape, shape)):
        if s_dim == 1 and a_dim != 1:
            arr = arr.sum(axis=i, keepdims=True)
    if arr.shape != shape:
        try:
            arr = arr.reshape(shape)
        except Exception:
            # Last resort: broadcast then sum
            arr = np.broadcast_to(arr, shape)
    return arr


# ------------------------- Autodiff Node -------------------------------
class Node:
    def __init__(self,
                 value: Any,
                 parents: Tuple['Node', ...] = (),
                 op: str = '',
                 name: Optional[str] = None,
                 is_param: bool = False):
        # normalize value to numpy array (float32)
        if isinstance(value, np.ndarray):
            self.value = value.astype(np.float32)
        else:
            self.value = np.array(value, dtype=np.float32)

        self.parents: Tuple[Node, ...] = tuple(parents)
        self.op = op
        self.name = name or op or f'Node_{id(self)}'
        self.is_param = is_param

        # public gradient view (dtype same as value)
        self.grad: Optional[np.ndarray] = None
        # internal accumulation buffer (float64) for stability if enabled
        self._grad_accum: Optional[np.ndarray] = None
        self._backward = lambda: None

    def __repr__(self):
        return f"Node(name={self.name!r}, op={self.op!r}, shape={self.value.shape})"

    # ---- arithmetic helpers ----
    def _ensure_node(self, other: Any) -> 'Node':
        return other if isinstance(other, Node) else Node(other, (), 'const')

    def _accumulate_grad(self, grad: np.ndarray):
        grad = np.asarray(grad)
        # reduce to the node shape first
        g = _sum_to_shape(grad, self.value.shape)
        if ENABLE_HIGH_PRECISION_ACCUMULATION:
            if self._grad_accum is None:
                self._grad_accum = np.zeros_like(self.value, dtype=np.float64)
            self._grad_accum += g.astype(np.float64)
            self.grad = self._grad_accum.astype(self.value.dtype)
        else:
            if self.grad is None:
                self.grad = np.zeros_like(self.value, dtype=self.value.dtype)
            self.grad += g.astype(self.value.dtype)

    # binary ops
    def __add__(self, other: Any) -> 'Node':
        other = self._ensure_node(other)
        out = Node(self.value + other.value, (self, other), 'add')
        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(out.grad)
            other._accumulate_grad(out.grad)
        out._backward = _backward
        return out

    def __radd__(self, other: Any) -> 'Node':
        return self.__add__(other)

    def __sub__(self, other: Any) -> 'Node':
        other = self._ensure_node(other)
        out = Node(self.value - other.value, (self, other), 'sub')
        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(out.grad)
            other._accumulate_grad(-out.grad)
        out._backward = _backward
        return out

    def __neg__(self) -> 'Node':
        out = Node(-self.value, (self,), 'neg')
        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(-out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other: Any) -> 'Node':
        other = self._ensure_node(other)
        out = Node(self.value * other.value, (self, other), 'mul')
        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(out.grad * other.value)
            other._accumulate_grad(out.grad * self.value)
        out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> 'Node':
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> 'Node':
        other = self._ensure_node(other)
        out = Node(self.value / other.value, (self, other), 'div')
        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(out.grad / other.value)
            other._accumulate_grad(-out.grad * self.value / (other.value ** 2))
        out._backward = _backward
        return out

    def __matmul__(self, other: Any) -> 'Node':
        other = self._ensure_node(other)
        out = Node(self.value @ other.value, (self, other), 'matmul')
        def _backward():
            if out.grad is None:
                return
            # out.grad shape (batch, out_features)
            self._accumulate_grad(out.grad @ other.value.T)
            other._accumulate_grad(self.value.T @ out.grad)
        out._backward = _backward
        return out

    def __pow__(self, power: float) -> 'Node':
        out = Node(self.value ** power, (self,), f'pow_{power}')
        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad((power * (self.value ** (power - 1))) * out.grad)
        out._backward = _backward
        return out

    # ---- elementwise activations ----
    def relu(self) -> 'Node':
        out = Node(np.maximum(0.0, self.value), (self,), 'relu')
        def _backward():
            if out.grad is None:
                return
            mask = (self.value > 0).astype(np.float32)
            self._accumulate_grad(mask * out.grad)
        out._backward = _backward
        return out

    def sigmoid(self) -> 'Node':
        x = np.clip(self.value, -500, 500)
        s = 1.0 / (1.0 + np.exp(-x))
        out = Node(s, (self,), 'sigmoid')
        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad((s * (1 - s)) * out.grad)
        out._backward = _backward
        return out

    def tanh(self) -> 'Node':
        t = np.tanh(self.value)
        out = Node(t, (self,), 'tanh')
        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad((1 - t * t) * out.grad)
        out._backward = _backward
        return out

    def exp(self) -> 'Node':
        x = np.clip(self.value, -100, 100)
        e = np.exp(x)
        out = Node(e, (self,), 'exp')
        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(e * out.grad)
        out._backward = _backward
        return out

    def log(self) -> 'Node':
        safe = np.maximum(self.value, 1e-15)
        l = np.log(safe)
        out = Node(l, (self,), 'log')
        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad((1.0 / safe) * out.grad)
        out._backward = _backward
        return out

    # ---- reductions and reshape ----
    def sum(self, axis=None, keepdims=False) -> 'Node':
        val = self.value.sum(axis=axis, keepdims=keepdims)
        out = Node(val, (self,), 'sum')
        def _backward():
            if out.grad is None:
                return
            g = out.grad
            if axis is None:
                g2 = np.broadcast_to(g, self.value.shape)
            else:
                if not keepdims:
                    if isinstance(axis, int):
                        g = np.expand_dims(g, axis)
                    else:
                        for ax in sorted(axis):
                            g = np.expand_dims(g, ax)
                g2 = np.broadcast_to(g, self.value.shape)
            self._accumulate_grad(g2)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False) -> 'Node':
        val = self.value.mean(axis=axis, keepdims=keepdims)
        out = Node(val, (self,), 'mean')
        def _backward():
            if out.grad is None:
                return
            if axis is None:
                denom = self.value.size
            else:
                if isinstance(axis, int):
                    denom = self.value.shape[axis]
                else:
                    denom = int(np.prod([self.value.shape[a] for a in axis]))
            g = out.grad / denom
            if axis is not None and not keepdims:
                if isinstance(axis, int):
                    g = np.expand_dims(g, axis)
                else:
                    for ax in sorted(axis):
                        g = np.expand_dims(g, ax)
            g2 = np.broadcast_to(g, self.value.shape)
            self._accumulate_grad(g2)
        out._backward = _backward
        return out

    def reshape(self, *shape) -> 'Node':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        val = self.value.reshape(*shape)
        out = Node(val, (self,), 'reshape')
        def _backward():
            if out.grad is None:
                return
            self._accumulate_grad(out.grad.reshape(self.value.shape))
        out._backward = _backward
        return out

    def transpose(self, axes=None) -> 'Node':
        val = self.value.transpose(axes) if axes is not None else self.value.T
        out = Node(val, (self,), 'transpose')
        def _backward():
            if out.grad is None:
                return
            if axes is None:
                g = out.grad.T
            else:
                inv = np.argsort(axes)
                g = out.grad.transpose(inv)
            self._accumulate_grad(g)
        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> 'Node':
        maxv = np.max(self.value, axis=axis, keepdims=True)
        exps = np.exp(self.value - maxv)
        probs = exps / np.sum(exps, axis=axis, keepdims=True)
        out = Node(probs, (self,), 'softmax')
        def _backward():
            if out.grad is None:
                return
            g = out.grad
            s = probs
            if s.ndim == 2:
                batch = s.shape[0]
                gin = np.empty_like(s)
                for i in range(batch):
                    si = s[i:i+1]
                    gi = g[i:i+1]
                    inner = (gi * si).sum(axis=1, keepdims=True)
                    gin[i:i+1] = (gi - inner) * si
            else:
                inner = (g * s).sum()
                gin = (g - inner) * s
            self._accumulate_grad(gin)
        out._backward = _backward
        return out

    # ---- topo/backward execution ----
    def _build_topo(self, visited=None) -> List['Node']:
        if visited is None:
            visited = set()
        topo: List[Node] = []
        def build(node: 'Node'):
            if id(node) in visited:
                return
            visited.add(id(node))
            for p in node.parents:
                build(p)
            topo.append(node)
        build(self)
        return topo

    def backward(self):
        topo = self._build_topo()
        # reset grads
        for n in topo:
            n.grad = None
            if ENABLE_HIGH_PRECISION_ACCUMULATION:
                n._grad_accum = np.zeros_like(n.value, dtype=np.float64)
        # seed gradient
        self.grad = np.ones_like(self.value, dtype=self.value.dtype)
        if ENABLE_HIGH_PRECISION_ACCUMULATION:
            self._grad_accum = self.grad.astype(np.float64)
        # execute
        for node in reversed(topo):
            try:
                node._backward()
            except Exception as e:
                print(f"Warning: backward failed for node {node}: {e}")

    def zero_grad(self):
        self.grad = None
        self._grad_accum = None


# ------------------------- Parameter & Module --------------------------
class Parameter(Node):
    def __init__(self, value: Any, name: Optional[str] = None):
        super().__init__(value, (), 'param', name or 'param')
        self.is_param = True

class Module:
    def __init__(self):
        self._modules: Dict[str, Module] = {}
        self._parameters: Dict[str, Parameter] = {}

    def add_module(self, name: str, module: 'Module') -> 'Module':
        self._modules[name] = module
        return module

    def add_parameter(self, name: str, value: Any) -> Parameter:
        p = Parameter(value, name)
        self._parameters[name] = p
        return p

    def parameters(self) -> List[Parameter]:
        res: List[Parameter] = []
        for p in self._parameters.values():
            res.append(p)
        for m in self._modules.values():
            res.extend(m.parameters())
        return res

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        sd: Dict[str, Any] = {}
        for k, p in self._parameters.items():
            sd[k] = p.value.copy()
        for k, m in self._modules.items():
            sd[k] = m.state_dict()
        return sd

    def load_state_dict(self, sd: Dict[str, Any]):
        for k, v in sd.items():
            if k in self._parameters:
                self._parameters[k].value = np.array(v, dtype=np.float32)
            elif k in self._modules:
                self._modules[k].load_state_dict(v)

    def train(self):
        self._training = True

    def eval(self):
        self._training = False


# ------------------------- Layers --------------------------------------
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        limit = math.sqrt(6.0 / (in_features + out_features))
        w = np.random.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float32)
        self.weight = self.add_parameter('weight', w)
        self.bias = None
        if bias:
            b = np.zeros((out_features,), dtype=np.float32)
            self.bias = self.add_parameter('bias', b)

    def __call__(self, x: Node) -> Node:
        out_val = x.value @ self.weight.value
        if self.bias is not None:
            out_val = out_val + self.bias.value
        out = Node(out_val, (x, self.weight) if self.bias is None else (x, self.weight, self.bias), 'linear')
        def _backward():
            if out.grad is None:
                return
            g = out.grad
            # grad w.r.t input
            x._accumulate_grad(g @ self.weight.value.T)
            # grad w.r.t weight
            dw = x.value.T @ g
            self.weight._accumulate_grad(dw)
            if self.bias is not None:
                db = g.sum(axis=0)
                self.bias._accumulate_grad(db)
        out._backward = _backward
        return out

class ReLU(Module):
    def __call__(self, x: Node) -> Node:
        return x.relu()

class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis
    def __call__(self, x: Node) -> Node:
        return x.softmax(axis=self.axis)

class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        self.modules_list: List[Module] = list(modules)
        for idx, m in enumerate(self.modules_list):
            self.add_module(str(idx), m)
    def __call__(self, x: Node) -> Node:
        out = x
        for m in self.modules_list:
            out = m(out)
        return out


# ------------------------- Losses --------------------------------------
class MSELoss:
    def __call__(self, preds: Node, targets: np.ndarray) -> Node:
        targets = np.asarray(targets, dtype=np.float32)
        diff = preds.value - targets
        val = np.mean(diff ** 2)
        out = Node(val, (preds,), 'mse')
        def _backward():
            if out.grad is None:
                return
            grad = (2.0 * diff / diff.size) * out.grad
            preds._accumulate_grad(grad)
        out._backward = _backward
        return out

class SoftmaxCrossEntropyLoss:
    def __call__(self, logits: Node, labels: np.ndarray) -> Node:
        labels = np.asarray(labels, dtype=np.int64)
        maxv = np.max(logits.value, axis=1, keepdims=True)
        exps = np.exp(logits.value - maxv)
        soft = exps / np.sum(exps, axis=1, keepdims=True)
        batch = logits.value.shape[0]
        probs = soft[np.arange(batch), labels]
        loss_vals = -np.log(probs + 1e-15)
        out = Node(np.mean(loss_vals), (logits,), 'softmax_ce')
        def _backward():
            if out.grad is None:
                return
            g = soft.copy()
            g[np.arange(batch), labels] -= 1
            g = g / batch
            logits._accumulate_grad(g * out.grad)
        out._backward = _backward
        return out


# ------------------------- Optimizers ----------------------------------
class OptimizerBase:
    def __init__(self, params: Iterable[Parameter], lr: float = 1e-3, clip_grad_norm: Optional[float] = None):
        self.params = list(params)
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError

class SGD(OptimizerBase):
    def __init__(self, params: Iterable[Parameter], lr: float = 1e-3, momentum: float = 0.0, weight_decay: float = 0.0, clip_grad_norm: Optional[float] = None):
        super().__init__(params, lr, clip_grad_norm)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.state: Dict[int, Dict[str, np.ndarray]] = {id(p): {'momentum': np.zeros_like(p.value, dtype=np.float64)} for p in self.params}

    def step(self):
        if self.clip_grad_norm is not None:
            clip_grad_norm_(self.params, self.clip_grad_norm)
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.astype(np.float64)
            if self.weight_decay:
                g = g + self.weight_decay * p.value.astype(np.float64)
            st = self.state[id(p)]
            m = st['momentum']
            m[:] = self.momentum * m + g
            update = self.lr * m
            p.value = (p.value.astype(np.float64) - update).astype(p.value.dtype)

class Adam(OptimizerBase):
    def __init__(self, params: Iterable[Parameter], lr: float = 1e-3, betas: Tuple[float,float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0, clip_grad_norm: Optional[float] = None):
        super().__init__(params, lr, clip_grad_norm)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state: Dict[int, Dict[str, Any]] = {}
        for p in self.params:
            self.state[id(p)] = {'step': 0, 'm': np.zeros_like(p.value, dtype=np.float64), 'v': np.zeros_like(p.value, dtype=np.float64)}

    def step(self):
        if self.clip_grad_norm is not None:
            clip_grad_norm_(self.params, self.clip_grad_norm)
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.astype(np.float64)
            if self.weight_decay:
                g = g + self.weight_decay * p.value.astype(np.float64)
            st = self.state[id(p)]
            st['step'] += 1
            m = st['m']; v = st['v']
            b1, b2 = self.betas
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * (g * g)
            m_hat = m / (1 - b1 ** st['step'])
            v_hat = v / (1 - b2 ** st['step'])
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.value = (p.value.astype(np.float64) - update).astype(p.value.dtype)


def clip_grad_norm_(params: Iterable[Parameter], max_norm: float, eps: float = 1e-6) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(np.sum(np.square(p.grad.astype(np.float64))))
    total = float(np.sqrt(total))
    if total == 0:
        return total
    clip_coef = max_norm / (total + eps)
    if clip_coef < 1.0:
        for p in params:
            if p.grad is None:
                continue
            p.grad = (p.grad.astype(np.float64) * clip_coef).astype(p.grad.dtype)
            if p._grad_accum is not None:
                p._grad_accum = (p._grad_accum * clip_coef).astype(np.float64)
    return total


# ------------------------- Data utilities --------------------------------
class NumpyDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y)
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataLoader:
    def __init__(self, dataset: NumpyDataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __iter__(self):
        idxs = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idxs)
        for i in range(0, len(idxs), self.batch_size):
            batch_idx = idxs[i:i+self.batch_size]
            Xb = np.stack([self.dataset[j][0] for j in batch_idx], axis=0)
            yb = np.stack([self.dataset[j][1] for j in batch_idx], axis=0)
            yield Xb, yb


# ------------------------- Training helpers --------------------------------
def save_model(module: Module, path: str):
    sd = module.state_dict()
    with open(path, 'wb') as f:
        pickle.dump(sd, f)

def load_model(module: Module, path: str):
    with open(path, 'rb') as f:
        sd = pickle.load(f)
    module.load_state_dict(sd)


def train_epoch(model: Module, dataloader: DataLoader, loss_fn, optimizer: OptimizerBase, verbose: bool = False):
    model.train()
    total_loss = 0.0
    batches = 0
    for Xb, yb in dataloader:
        xb = Node(Xb)
        preds = model(xb)
        loss_node = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss_node.backward()
        optimizer.step()
        total_loss += float(loss_node.value)
        batches += 1
        if verbose:
            print(f"batch {batches}, loss={float(loss_node.value):.6f}")
    return total_loss / max(1, batches)


def evaluate(model: Module, dataloader: DataLoader, loss_fn) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    for Xb, yb in dataloader:
        xb = Node(Xb)
        preds = model(xb)
        loss_node = loss_fn(preds, yb)
        total_loss += float(loss_node.value)
        batches += 1
    return total_loss / max(1, batches)


# ------------------------- Demo / Quick smoke -------------------------------
if __name__ == '__main__':
    set_seed(42)
    # toy regression
    N, D_in, D_out = 200, 3, 1
    X = np.random.randn(N, D_in).astype(np.float32)
    true_w = np.array([[2.0], [-1.0], [0.5]], dtype=np.float32)
    y = (X @ true_w + 0.1 * np.random.randn(N, 1)).astype(np.float32)

    ds = NumpyDataset(X, y)
    dl = DataLoader(ds, batch_size=32)

    model = Sequential(Linear(D_in, 16), ReLU(), Linear(16, D_out))
    opt = Adam(model.parameters(), lr=1e-2)
    loss_fn = MSELoss()

    for epoch in range(5):
        tr = train_epoch(model, dl, loss_fn, opt, verbose=True)
        print(f"epoch {epoch}, train_loss={tr:.6f}")

    save_model(model, 'manus_example.pkl')
