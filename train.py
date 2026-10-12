import numpy as np
import matplotlib.pyplot as plt
from dataset import load_and_prep_mnist
from engine import initialize_parameters_deep, forward_propagation_deep, compute_cost, backward_propagation_deep
from activations import get_predictions, get_accuracy

def create_mini_batches(X, Y, batch_size=128):
    m = X.shape[0]
    mini_batches = []
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]
    num_complete_batches = m // batch_size
    
    for k in range(num_complete_batches):
        batch_X = shuffled_X[k * batch_size : (k + 1) * batch_size, :]
        batch_Y = shuffled_Y[k * batch_size : (k + 1) * batch_size, :]
        mini_batches.append((batch_X, batch_Y))
        
    if m % batch_size != 0:
        batch_X = shuffled_X[num_complete_batches * batch_size : , :]
        batch_Y = shuffled_Y[num_complete_batches * batch_size : , :]
        mini_batches.append((batch_X, batch_Y))
        
    return mini_batches

def train_L_layer_model(X_train, Y_train_encoded, layer_dims, learning_rate=0.1, epochs=50, batch_size=128):
    # 1. Initialize dynamic dictionary
    parameters = initialize_parameters_deep(layer_dims)
    cost_history = []
    L = len(parameters) // 2  # Divides by 2 because dictionary has W and b for each layer
    
    for epoch in range(epochs):
        epoch_cost = 0
        batches = create_mini_batches(X_train, Y_train_encoded, batch_size)
        
        for batch_X, batch_Y in batches:
            # 2. Forward Pass (returns the final probability AL, and the breadcrumbs caches)
            AL, caches = forward_propagation_deep(batch_X, parameters)
            
            # 3. Compute Loss
            cost = compute_cost(batch_Y, AL)
            epoch_cost += cost
            
            # 4. Backward Pass (returns dictionary of gradients)
            grads = backward_propagation_deep(AL, batch_Y, caches)
            
            # 5. Update Weights dynamically using a loop
            for l in range(1, L + 1):
                parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
                parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
            
        average_epoch_cost = epoch_cost / len(batches)
        cost_history.append(average_epoch_cost)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch} | Average Cost: {average_epoch_cost:.4f}")
            
    return parameters, cost_history

if __name__ == "__main__":
    X_train, Y_train_encoded, Y_train_raw, X_test, Y_test_encoded, Y_test_raw = load_and_prep_mnist()
    
    # Try [784, 128, 64, 10] or change it to [784, 256, 128, 64, 10] to push the engine (and your computer) a bit more lmao
    layer_dims = [784, 128, 64, 10]
    
    print(f"\nTraining {len(layer_dims)-1}-Layer Network...")
    parameters, history = train_L_layer_model(
        X_train, Y_train_encoded, layer_dims, 
        learning_rate=0.1, epochs=50, batch_size=128
    )
    
    print("\nEvaluating on Test Data...")
    AL_test, _ = forward_propagation_deep(X_test, parameters)
    test_predictions = get_predictions(AL_test)
    test_accuracy = get_accuracy(test_predictions, Y_test_raw)
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")
    
    # Saving the dictionary for release of drawing ui
    np.save("trained_parameters.npy", parameters)
    print("\nModel weights saved to trained_parameters.npy")