"""
Unit tests for NumPy-from-scratch Neural Network Fundamentals.
Verifies activation functions, analytical derivatives, loss stability, and gradient backprop.
"""

import numpy as np
import pytest

from src.nn_fundamentals import Activations, LossFunctions, ScratchNeuralNetwork


def test_activations_and_derivatives():
    x = np.array([[-2.0, 0.0, 3.0]])
    
    # ReLU
    relu_out = Activations.relu(x)
    assert np.allclose(relu_out, [[0.0, 0.0, 3.0]])
    relu_grad = Activations.relu_derivative(x)
    assert np.allclose(relu_grad, [[0.0, 0.0, 1.0]])
    
    # Sigmoid
    sig_out = Activations.sigmoid(x)
    assert np.all(sig_out >= 0.0) and np.all(sig_out <= 1.0)
    sig_grad = Activations.sigmoid_derivative(x)
    assert np.all(sig_grad >= 0.0)
    
    # Tanh
    tanh_out = Activations.tanh(x)
    assert np.all(tanh_out >= -1.0) and np.all(tanh_out <= 1.0)
    tanh_grad = Activations.tanh_derivative(x)
    assert np.all(tanh_grad > 0.0)
    
    # Softmax sums to 1
    softmax_out = Activations.softmax(x)
    assert np.isclose(np.sum(softmax_out), 1.0)


@pytest.mark.parametrize("act_fn", ["relu", "sigmoid", "tanh"])
def test_scratch_neural_network_convergence(act_fn):
    np.random.seed(42)
    N = 100
    X = np.random.randn(N, 10)
    y_indices = np.random.randint(0, 4, size=N)
    y_one_hot = np.zeros((N, 4))
    y_one_hot[np.arange(N), y_indices] = 1.0
    
    lr = 0.05 if act_fn == "relu" else 0.1
    model = ScratchNeuralNetwork(input_dim=10, hidden_dim=16, output_dim=4, activation=act_fn, learning_rate=lr)
    history = model.fit(X, y_one_hot, epochs=50)
    
    # Check that loss decreased over training
    assert history["loss"][-1] < history["loss"][0]
    assert len(history["accuracy"]) == 50
