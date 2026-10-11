import numpy as np
import matplotlib.pyplot as plt
from dataset import load_and_prep_mnist
from engine import initialize_parameters, forward_propagation, compute_cost, backward_propagation
from activations import get_predictions, get_accuracy

def create_mini_batches(X, Y, batch_size=128):
    """Shuffles the dataset and slices it into perfectly sized batches."""
    m = X.shape[0]
    mini_batches = []
    
    # Shuffle X and Y synchronously
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]
    
    num_complete_batches = m // batch_size
    
    for k in range(num_complete_batches):
        batch_X = shuffled_X[k * batch_size : (k + 1) * batch_size, :]
        batch_Y = shuffled_Y[k * batch_size : (k + 1) * batch_size, :]
        mini_batches.append((batch_X, batch_Y))
        
    # Catch any leftover images if 60,000 doesn't divide perfectly by the batch size
    if m % batch_size != 0:
        batch_X = shuffled_X[num_complete_batches * batch_size : , :]
        batch_Y = shuffled_Y[num_complete_batches * batch_size : , :]
        mini_batches.append((batch_X, batch_Y))
        
    return mini_batches

def train_model(X_train, Y_train_encoded, learning_rate=0.1, epochs=50, batch_size=128, hidden_units=64):
    n_features = X_train.shape[1]
    
    # Output classes is locked to 10 for MNIST
    W1, b1, W2, b2 = initialize_parameters(n_features, hidden_units, output_classes=10)
    cost_history = []
    
    for epoch in range(epochs):
        epoch_cost = 0
        batches = create_mini_batches(X_train, Y_train_encoded, batch_size)
        
        for batch_X, batch_Y in batches:
            # 1. Forward Pass
            Z1, A1, A2 = forward_propagation(batch_X, W1, b1, W2, b2)
            
            # 2. Compute Loss
            cost = compute_cost(batch_Y, A2)
            epoch_cost += cost
            
            # 3. Backward Pass
            dW1, db1, dW2, db2 = backward_propagation(batch_X, batch_Y, Z1, A1, A2, W2)
            
            # 4. Update Weights
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            
        # Store average cost for the entire epoch
        average_epoch_cost = epoch_cost / len(batches)
        cost_history.append(average_epoch_cost)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch} | Average Cost: {average_epoch_cost:.4f}")
            
    return W1, b1, W2, b2, cost_history

if __name__ == "__main__":
    # 1. Load Data
    X_train, Y_train_encoded, Y_train_raw, X_test, Y_test_encoded, Y_test_raw = load_and_prep_mnist()
    
    # 2. Train Engine
    print("\nTraining RawNet on 60,000 images...")
    W1, b1, W2, b2, history = train_model(
        X_train, Y_train_encoded, 
        learning_rate=0.1, 
        epochs=50, 
        batch_size=128, 
        hidden_units=64
    )
    
    # 3. Evaluate on unseen Test Data
    print("\nEvaluating on 10,000 unseen test images...")
    _, _, A2_test = forward_propagation(X_test, W1, b1, W2, b2)
    test_predictions = get_predictions(A2_test)
    test_accuracy = get_accuracy(test_predictions, Y_test_raw)
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")
    
    # 4. Plot Learning Curve
    plt.plot(history, color='blue', linewidth=2)
    plt.title('RawNet Learning Curve (Categorical Cross-Entropy)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()