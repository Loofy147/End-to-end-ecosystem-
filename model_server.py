from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import time
import json

# ==============================================================================
#  القسم 1: إطار العمل الكامل للتعلم العميق
# ==============================================================================

def _sum_to_shape(grad, shape):
    while grad.ndim > len(shape): grad = grad.sum(axis=0)
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, shape)):
        if t_dim == 1 and g_dim != 1: grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)

class Node:
    def __init__(self, value, parents=(), op=''):
        self.value = np.array(value, dtype=float); self.parents = parents; self.op = op
        self.grad = None; self._backward = lambda: None
    def _ensure(self, other):
        return other if isinstance(other, Node) else Node(np.array(other, dtype=float))
    def __add__(self, other):
        other = self._ensure(other); out = Node(self.value + other.value, (self, other), '+')
        def _backward():
            if out.grad is None: return
            self.grad += _sum_to_shape(out.grad, self.value.shape)
            other.grad += _sum_to_shape(out.grad, other.value.shape)
        out._backward = _backward; return out
    def __mul__(self, other):
        other = self._ensure(other); out = Node(self.value * other.value, (self, other), '*')
        def _backward():
            if out.grad is None: return
            self.grad += _sum_to_shape(out.grad * other.value, self.value.shape)
            other.grad += _sum_to_shape(out.grad * self.value, other.value.shape)
        out._backward = _backward; return out
    def __matmul__(self, other):
        other = self._ensure(other); out = Node(self.value @ other.value, (self, other), '@')
        def _backward():
            if out.grad is None: return
            A, B, G = self.value, other.value, out.grad
            self.grad += _sum_to_shape(G @ B.T, self.value.shape)
            other.grad += _sum_to_shape(A.T @ G, other.value.shape)
        out._backward = _backward; return out
    def relu(self):
        out = Node(np.maximum(0, self.value), (self,), 'relu')
        def _backward():
            if out.grad is None: return
            self.grad += (self.value > 0) * out.grad
        out._backward = _backward; return out
    def sum(self, axis=None, keepdims=False):
        out = Node(self.value.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            if out.grad is None: return
            grad = out.grad
            if not keepdims and axis is not None: grad = np.expand_dims(out.grad, axis)
            self.grad += np.broadcast_to(grad, self.value.shape)
        out._backward = _backward; return out
    def exp(self):
        out = Node(np.exp(self.value), (self,), 'exp')
        def _backward():
            if out.grad is None: return
            self.grad += out.value * out.grad
        out._backward = _backward; return out
    def softmax(self, axis=-1):
        shifted_self = self + (self.value.max(axis=axis, keepdims=True) * -1)
        exp_x = shifted_self.exp(); sum_exp_x = exp_x.sum(axis=axis, keepdims=True)
        return exp_x * (sum_exp_x**-1)
    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for p in v.parents: build_topo(p)
                topo.append(v)
        build_topo(self)
        for v in topo: v.grad = np.zeros_like(v.value)
        self.grad = np.ones_like(self.value)
        for v in reversed(topo): v._backward()

class nn:
    class Module:
        def parameters(self): yield from []
        def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
        def zero_grad(self):
            for p in self.parameters(): p.grad = np.zeros_like(p.value)

    class Linear(Module):
        def __init__(self, in_features, out_features):
            self.weight = Node(np.random.randn(in_features, out_features) * 0.01)
            self.bias = Node(np.zeros(out_features))
        def forward(self, x): return x @ self.weight + self.bias
        def parameters(self): yield from [self.weight, self.bias]

    class ReLU(Module):
        def forward(self, x): return x.relu()

    class Flatten(Module):
        def __init__(self): self.cache = {}
        def forward(self, x_node):
            x = x_node.value
            self.cache['input_shape'] = x.shape
            output_node = Node(x.reshape(x.shape[0], -1), parents=(x_node,), op='flatten')
            def _backward():
                if output_node.grad is None: return
                x_node.grad += output_node.grad.reshape(self.cache['input_shape'])
            output_node._backward = _backward
            return output_node

    class Conv1d(Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1):
            self.in_channels, self.out_channels = in_channels, out_channels
            self.kernel_size, self.stride = kernel_size, stride
            w_shape = (out_channels, in_channels, kernel_size)
            self.weight = Node(np.random.randn(*w_shape) * np.sqrt(2. / (in_channels * kernel_size)))
            self.bias = Node(np.zeros(out_channels))
        
        def parameters(self): yield from [self.weight, self.bias]

        def forward(self, x_node):
            x = x_node.value
            N, C_in, L_in = x.shape
            L_out = (L_in - self.kernel_size) // self.stride + 1
            
            X_col = []
            for i in range(L_out):
                window = x[:, :, i*self.stride : i*self.stride+self.kernel_size]
                X_col.append(window.reshape(N, -1))
            X_col = np.array(X_col).transpose(1, 2, 0)
            
            W_row = self.weight.value.reshape(self.out_channels, -1)
            
            out = np.zeros((N, self.out_channels, L_out))
            for i in range(N):
                out[i] = W_row @ X_col[i] + self.bias.value.reshape(-1, 1)

            output_node = Node(out, parents=(x_node, self.weight, self.bias), op='conv1d')
            
            def _backward():
                if output_node.grad is None: return
                grad_w = np.zeros_like(self.weight.value)
                for i in range(N):
                    grad_w += (output_node.grad[i].T @ X_col[i].T).reshape(grad_w.shape)
                self.weight.grad += grad_w / N
                self.bias.grad += np.sum(output_node.grad, axis=(0, 2))
            output_node._backward = _backward
            return output_node

    class Sequential(Module):
        def __init__(self, *layers): self.layers = layers
        def forward(self, x):
            for layer in self.layers: x = layer(x)
            return x
        def parameters(self):
            for layer in self.layers: yield from layer.parameters()

# ==============================================================================
#  القسم 2: الخادم المحسّن
# ==============================================================================

app = Flask(__name__)

MODEL_CACHE = {}

def build_model_dynamically(config, input_shape):
    layers = []
    current_shape = input_shape
    
    for layer_conf in config['layers']:
        layer_type = layer_conf['type']
        if layer_type == 'conv1d':
            in_channels = current_shape[1]
            out_channels = layer_conf['out_channels']
            kernel_size = layer_conf['kernel_size']
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size))
            current_shape = (current_shape[0], out_channels, (current_shape[2] - kernel_size) + 1)
        elif layer_type == 'relu':
            layers.append(nn.ReLU())
        elif layer_type == 'flatten':
            layers.append(nn.Flatten())
            current_shape = (current_shape[0], current_shape[1] * current_shape[2])
        elif layer_type == 'linear':
            in_features = current_shape[1]
            out_features = layer_conf['out_features']
            layers.append(nn.Linear(in_features, out_features))
            current_shape = (current_shape[0], out_features)
            
    return nn.Sequential(*layers)

# --- NEW: دالة هندسة الميزات المتقدمة ---
def prepare_features(prices_df, window_size):
    df = prices_df.copy()
    
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_returns'].rolling(window=window_size).std()
    df['momentum'] = df['close'] / df['close'].rolling(window=window_size).mean() - 1
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    
    feature_names = ['log_returns', 'volatility', 'momentum', 'rsi']
    df_features = df[feature_names]
    
    df_scaled_features = (df_features - df_features.mean()) / (df_features.std() + 1e-8)
    
    X, y = [], []
    future_horizon = 3
    df['target'] = (df['close'].shift(-future_horizon) > df['close']).astype(int)
    df.dropna(inplace=True)
    
    for i in range(len(df_scaled_features) - window_size):
        X.append(df_scaled_features.iloc[i:i+window_size].values)
        y.append(df['target'].iloc[i+window_size-1])
        
    X = np.array(X).transpose(0, 2, 1)
    
    return X, np.array(y)

@app.route('/train_and_predict_dynamic', methods=['POST'])
def dynamic_endpoint():
    start_time = time.time()
    
    data = request.get_json()
    config = data['model_config']
    
    config_key = json.dumps(config, sort_keys=True)
    if config_key in MODEL_CACHE:
        print("Server: Cache HIT! Returning stored predictions.")
        return jsonify({"predictions": MODEL_CACHE[config_key]})
    
    print(f"Server: Cache MISS. Training new model...")
    
    prices_data = data['prices']
    df = pd.DataFrame(prices_data)
    
    window_size = config.get('window_size', 15)
    learning_rate = config.get('learning_rate', 0.01)
    epochs = config.get('epochs', 50)
    
    X_train, y_train = prepare_features(df, window_size)
    if len(X_train) == 0: return jsonify({"predictions": np.full(len(prices_data['close']), 0.5).tolist()})
    
    num_features = X_train.shape[1]
    config['layers'][0]['in_channels'] = num_features
    
    model = build_model_dynamically(config, input_shape=X_train.shape)
    
    params = list(model.parameters())
    X_node = Node(X_train)
    for epoch in range(epochs):
        logits = model(X_node)
        probabilities = logits.softmax()
        batch_size = probabilities.value.shape[0]
        true_class_probs = Node(probabilities.value[np.arange(batch_size), y_train])
        clipped_probs = Node(np.clip(true_class_probs.value, 1e-10, 1.0))
        log_loss = Node(np.log(clipped_probs.value) * -1)
        loss = log_loss.sum() * (1 / batch_size)
        model.zero_grad()
        loss.backward()
        for p in params: p.value -= learning_rate * p.grad
        
    all_predictions = np.full(len(prices_data['close']), 0.5)
    for i in range(len(prices_data['close']) - window_size):
        window_df = df.iloc[i:i+window_size]
        X_test_single, _ = prepare_features(window_df, window_size)
        
        if X_test_single.shape[0] == 0: continue

        input_tensor = Node(X_test_single)
        logits = model(input_tensor)
        probs = logits.softmax().value
        all_predictions[i + window_size] = probs[0][1]
        
    end_time = time.time()
    print(f"Server: Dynamic request processed in {end_time - start_time:.2f}s.")
    
    MODEL_CACHE[config_key] = all_predictions.tolist()
    
    return jsonify({"predictions": all_predictions.tolist()})

if __name__ == '__main__':
    print("Starting Python model server v4.0 (Feature Engineering) at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)


