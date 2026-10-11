import numpy as np
# Use tensorflow.keras.datasets if this doesn't work
from keras.datasets import mnist

def one_hot_encode(Y, num_classes=10):
    m = Y.shape[0]
    one_hot = np.zeros((m, num_classes))

    # Place 1 in the correct class index for every sample
    one_hot[np.arange(m), Y] = 1.0
    return one_hot

def load_and_prep_mnist():
    print("Downloading and preparing MNIST dataset...")
    
    # 1. Fetch raw data
    (X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw) = mnist.load_data()
    
    # 2. Flatten the 28x28 images into 784-pixel rows
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)
    
    # 3. Normalize pixel values to range 0.0 - 1.0
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # 4. One-Hot Encode the labels 
    Y_train_encoded = one_hot_encode(Y_train_raw, num_classes=10)
    Y_test_encoded = one_hot_encode(Y_test_raw, num_classes=10)
    
    print(f"Training data shape (X): {X_train.shape}")
    print(f"Training labels shape (Y_encoded): {Y_train_encoded.shape}")
    
    # Returning raw Y as well so I can easily check the final accuracy (Don't forget to use the encoded Y for training)
    return X_train, Y_train_encoded, Y_train_raw, X_test, Y_test_encoded, Y_test_raw