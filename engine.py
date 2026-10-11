import numpy as np
from activations import ReLU, dReLU, Softmax

def initialize_parameters(n_features, hidden_units, output_classes=10):

    # Layer 1 HE initialization
    W1 = np.random.randn(n_features, hidden_units) * np.sqrt(2. / n_features)
    b1 = np.zeros((1, hidden_units))

    # Layer 2 normal initialization
    W2 = np.random.randn(hidden_units, output_classes) * 0.01
    b2 = np.zeros((1, output_classes))
    
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    # First layer
    Z1 = np.dot(X, W1) + b1
    A1 = ReLU(Z1)

    # Second Layer
    Z2 = np.dot(A1, W2) + b2
    A2 = Softmax(Z2)

    return Z1, A1, A2

def compute_cost(A2, Y_one_hot):
    m = Y_one_hot.shape[0]
    epsilon = 1e-15  # Small value to avoid log(0)
    A2 = np.clip(A2, epsilon, 1 - epsilon)  # Clip A2 to avoid log(0)
    cost = -np.sum(Y_one_hot * np.log(A2)) / m

    return cost

def backward_propagation(X, Y_one_hot, Z1, A1, A2, W2):
    m = X.shape[0]

    # Output layer gradients
    dZ2 = A2 - Y_one_hot
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer gradients
    dZ1 = np.dot(dZ2, W2.T) * dReLU(Z1)
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2