import gradio as gr
import numpy as np
from PIL import Image
from engine import forward_propagation_deep

# 1. Load the trained L-layer matrices into RAM once when the server starts
parameters = np.load("trained_parameters.npy", allow_pickle=True).item()

def predict_digit(img_dict):
    """Processes the canvas drawing and runs the math engine."""
    if img_dict is None:
        return "Please draw a digit."
    
    # Gradio 4.x Sketchpad returns a dict containing layers. We grab the composite image.
    img_array = img_dict["composite"]
    
    # Convert to a PIL Image (Grayscale) to easily resize down to 28x28
    img_pil = Image.fromarray(img_array).convert('L')
    img_resized = img_pil.resize((28, 28))
    
    # Convert back to a NumPy matrix
    img_matrix = np.array(img_resized)
    
    # MNIST was trained on white digits with a black background.
    # If the canvas background is white, we mathematically invert the colors.
    if img_matrix[0, 0] > 127:
        img_matrix = 255 - img_matrix
        
    # Flatten into the 784-column array and scale pixels to 0.0-1.0
    X_input = img_matrix.reshape(1, 784) / 255.0
    
    # 2. Push the drawing through the dynamic L-layer engine
    AL, _ = forward_propagation_deep(X_input, parameters)
    
    # 3. Extract the prediction
    prediction = np.argmax(AL, axis=1)[0]
    confidence = np.max(AL) * 100
    
    return f"Prediction: {prediction} | Confidence: {confidence:.2f}%"

# 4. Build the web interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Draw a digit (0-9)", type="numpy"),
    outputs=gr.Text(label="RawNet Engine Output"),
    title="ZeroGrad Engine: Live MNIST Recognition",
    description="Draw a digit in the box below. This uses a custom pure-NumPy Multi-Layer Perceptron built from scratch.",
    theme="default"
)

if __name__ == "__main__":
    demo.launch()