import numpy as np
from activations import ReLU, dReLU, Softmax

def initialize_parameters_deep(layer_dims):
    """
    Takes a list like [784, 128, 64, 10] and dynamically builds the matrices.
    """
    parameters = {}
    L = len(layer_dims)            

    for l in range(1, L):
        # Use He Initialization for hidden layers, standard * 0.01 for the final Softmax layer
        if l == L - 1:
            parameters['W' + str(l)] = np.random.randn(layer_dims[l-1], layer_dims[l]) * 0.01
        else:
            parameters['W' + str(l)] = np.random.randn(layer_dims[l-1], layer_dims[l]) * np.sqrt(2. / layer_dims[l-1])
            
        parameters['b' + str(l)] = np.zeros((1, layer_dims[l]))
        
    return parameters

def forward_propagation_deep(X, parameters):
    """
    Loops through the dictionary to push data through any number of layers.
    Saves every calculation in a 'caches' dictionary for backpropagation.
    """
    caches = {}
    A = X
    L = len(parameters) // 2  # Number of layers (W and b for each)
    
    # Hidden layers: 1 to L-1 (ReLU)
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        
        Z = np.dot(A_prev, W) + b
        A = ReLU(Z)
        
        # Store for the chain rule later
        caches['A' + str(l-1)] = A_prev
        caches['Z' + str(l)] = Z
        caches['W' + str(l)] = W

    # Output layer: L (Softmax)
    W_last = parameters['W' + str(L)]
    b_last = parameters['b' + str(L)]
    Z_last = np.dot(A, W_last) + b_last
    AL = Softmax(Z_last)
    
    caches['A' + str(L-1)] = A
    caches['Z' + str(L)] = Z_last
    caches['W' + str(L)] = W_last
    
    return AL, caches

def compute_cost(Y_one_hot, AL):
    m = Y_one_hot.shape[0]
    epsilon = 1e-15
    AL = np.clip(AL, epsilon, 1 - epsilon)
    cost = -(1 / m) * np.sum(Y_one_hot * np.log(AL))
    return cost

def backward_propagation_deep(AL, Y_one_hot, caches):
    """
    Calculates gradients by looping backwards from layer L down to layer 1.
    """
    grads = {}
    L = len(caches) // 3  # We stored A_prev, Z, and W for each layer
    m = AL.shape[0]
    
    # 1. Start with the Output Layer Error
    dZ = AL - Y_one_hot
    
    # 2. Step backward through the network
    for l in reversed(range(1, L + 1)):
        A_prev = caches['A' + str(l-1)]
        W = caches['W' + str(l)]
        
        # Calculate gradients for the current layer
        dW = (1 / m) * np.dot(A_prev.T, dZ)
        db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
        
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db
        
        # If we are not at the first layer yet, calculate the error to pass backwards
        if l > 1:
            Z_prev = caches['Z' + str(l-1)]
            dZ = np.dot(dZ, W.T) * dReLU(Z_prev)
            
    return grads