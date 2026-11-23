"""
Neural Network Fundamentals from Scratch using NumPy.
Covers:
- Activation functions: ReLU, Sigmoid, Tanh, Softmax and their analytical derivatives
- Forward propagation
- Numerical stability with log-sum-exp trick for Cross-Entropy loss
- Analytical backpropagation (chain rule)
- Optimization: SGD with Momentum and L2 Weight Decay
"""

from typing import Dict, Tuple
import numpy as np


class Activations:
    """Activation functions and their analytical derivatives."""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)
        
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float64)
        
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # Numerically clipped sigmoid to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))
        
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = Activations.sigmoid(x)
        return s * (1.0 - s)
        
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
        
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(x) ** 2
        
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax using the max subtraction trick."""
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class LossFunctions:
    """Loss functions and gradient computation."""
    
    @staticmethod
    def cross_entropy(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
        """
        Calculates categorical cross-entropy.
        Args:
            probs: Predicted probabilities of shape (batch_size, num_classes)
            y_true: One-hot encoded labels of shape (batch_size, num_classes)
        """
        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        loss = -np.sum(y_true * np.log(probs_clipped)) / y_true.shape[0]
        return float(loss)


class ScratchNeuralNetwork:
    """
    A 2-layer Multi-Layer Perceptron implemented purely in NumPy.
    Supports modular activation functions (ReLU, Sigmoid, Tanh).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "relu",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        seed: int = 42
    ):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_name = activation.lower()
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # He / Xavier normal initialization
        if self.activation_name == "relu":
            std1 = np.sqrt(2.0 / input_dim)
        else:
            std1 = np.sqrt(1.0 / input_dim)
        std2 = np.sqrt(2.0 / hidden_dim)
        
        self.W1 = np.random.randn(input_dim, hidden_dim) * std1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * std2
        self.b2 = np.zeros((1, output_dim))
        
        # Momentum velocity buffers
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        
        # Cache for backprop
        self.cache: Dict[str, np.ndarray] = {}
        
    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            return Activations.relu(z)
        elif self.activation_name == "sigmoid":
            return Activations.sigmoid(z)
        elif self.activation_name == "tanh":
            return Activations.tanh(z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
            
    def _apply_activation_derivative(self, z: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            return Activations.relu_derivative(z)
        elif self.activation_name == "sigmoid":
            return Activations.sigmoid_derivative(z)
        elif self.activation_name == "tanh":
            return Activations.tanh_derivative(z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation.
        Z1 = X @ W1 + b1
        A1 = activation(Z1)
        Z2 = A1 @ W2 + b2
        Probs = Softmax(Z2)
        """
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._apply_activation(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        probs = Activations.softmax(Z2)
        
        self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "probs": probs}
        return probs

    def backward(self, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward propagation using analytical chain rule gradients.
        dLoss/dZ2 = (Probs - y_true) / N
        """
        N = y_true.shape[0]
        probs = self.cache["probs"]
        A1 = self.cache["A1"]
        Z1 = self.cache["Z1"]
        X = self.cache["X"]
        
        # Output layer gradient (Softmax + Cross Entropy simplifies to: probs - y)
        dZ2 = (probs - y_true) / N
        dW2 = np.dot(A1.T, dZ2) + self.weight_decay * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradient
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._apply_activation_derivative(Z1)
        dW1 = np.dot(X.T, dZ1) + self.weight_decay * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        """SGD with Momentum parameter update."""
        self.v_W1 = self.momentum * self.v_W1 + self.lr * grads["dW1"]
        self.v_b1 = self.momentum * self.v_b1 + self.lr * grads["db1"]
        self.v_W2 = self.momentum * self.v_W2 + self.lr * grads["dW2"]
        self.v_b2 = self.momentum * self.v_b2 + self.lr * grads["db2"]
        
        self.W1 -= self.v_W1
        self.b1 -= self.v_b1
        self.W2 -= self.v_W2
        self.b2 -= self.v_b2

    def fit(self, X: np.ndarray, y_true: np.ndarray, epochs: int = 100) -> Dict[str, list]:
        """Training loop over synthetic/tabular representations."""
        history = {"loss": [], "accuracy": []}
        for epoch in range(epochs):
            probs = self.forward(X)
            loss = LossFunctions.cross_entropy(probs, y_true)
            grads = self.backward(y_true)
            self.step(grads)
            
            preds = np.argmax(probs, axis=1)
            targets = np.argmax(y_true, axis=1)
            acc = float(np.mean(preds == targets))
            
            history["loss"].append(loss)
            history["accuracy"].append(acc)
            
        return history
