import numpy as np

def ReLU(z):
    return np.maximum(0, z)

def dReLU(z):
    return (z > 0).astype(float)

def Softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def get_predictions(A2):
    return np.argmax(A2, axis=1)

def get_accuracy(predictions, Y_raw):
    return np.mean(predictions == Y_raw) * 100