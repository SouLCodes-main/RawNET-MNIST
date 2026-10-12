# RawNET-MNIST 🧠

A pure-NumPy, functional-style neural network built entirely from scratch to classify handwritten digits. This project demonstrates the core calculus of deep learning by implementing a custom mathematics engine, He Initialization, ReLU/Softmax activations, and mini-batch gradient descent, all without relying on auto-differentiation frameworks like PyTorch or TensorFlow.

[![Accuracy](https://img.shields.io/badge/Accuracy-98.02%25-success)](#)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![NumPy Version](https://img.shields.io/badge/NumPy-2.5.4-orange.svg)](https://numpy.org/)

---

## Table of Contents 📚

- [Project Title & Badges](#rawnet-mnist-)
- [Description](#description-)
- [Features](#features-)
- [Tech Stack](#tech-stack-)
- [Installation](#installation-)
- [Usage](#usage-)
- [Project Structure](#project-structure-)
- [Contributing](#contributing-)
- [License](#license-)
- [Important Links](#important-links-)
- [Footer](#footer-)

---

## Description 📝

RawNET-MNIST is an educational project that dives deep into the fundamental principles of neural networks. It is a complete implementation of a feedforward neural network using only NumPy, eschewing high-level deep learning libraries. 

Originally built as a 2-layer network, it has been engineered into a **Dynamic L-Layer Architecture**, capable of generating and training any sequence of hidden layers dynamically. The project also includes a live interactive web canvas, allowing users to draw digits and test the pure-math engine in real-time.

---

## Features ✨

- **Dynamic L-Layer Architecture:** Configure any number of layers and nodes dynamically (e.g., `[784, 128, 64, 10]`) using intelligent dictionary state management.
- **Interactive Web UI:** Features a local Gradio web server with a drawing canvas to test the model's inference in real-time.
- **Pure NumPy Implementation:** No external deep learning frameworks (PyTorch, TensorFlow) used for core network operations.
- **Custom Mathematics Engine:** Implements all necessary mathematical operations from scratch.
- **He Initialization:** Utilizes He initialization for weights, mathematically tuned for ReLU activation functions.
- **Activation Functions:** Includes ReLU (Rectified Linear Unit) for hidden layers and a numerically stable Softmax for the output layer.
- **Mini-Batch Gradient Descent:** Implements efficient training using mini-batches and Categorical Cross-Entropy loss.
- **Model Serialization:** Saves trained weights as `.npy` binaries for instant UI loading without retraining.

---

## Tech Stack 💻

- **Language:** Python
- **Core Library:** NumPy
- **UI Canvas:** Gradio, Pillow (PIL)
- **Data Loading:** Keras Datasets (via `keras` package, used strictly for downloading data)
- **Plotting:** Matplotlib
- **Build System:** uv_build

---

## Installation 🛠️

This project is designed to be run with Python 3.13+. The `pyproject.toml` file lists the necessary dependencies. You can install them using `pip` or `uv`.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SouLCodes-main/RawNET-MNIST.git](https://github.com/SouLCodes-main/RawNET-MNIST.git)
    cd RawNET-MNIST
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.

    Using `pip` for direct execution:
    ```bash
    pip install numpy matplotlib keras gradio pillow
    ```

    Using `uv` (if you have it installed):
    ```bash
    uv pip install . 
    ```

---

## Usage 🚀

The project operates in two phases: Training the engine, and testing it via the UI.

### 1. Train the Model
Running `train.py` will load the data, dynamically initialize the L-layer architecture, execute the mini-batch training loop, output the test accuracy, and save the serialized weights to your disk.

```bash
python train.py
```

### 2. Launch the Interactive Canvas
Once trained, launch the UI script to open a local web server. You can draw numbers on the canvas and watch the custom NumPy engine predict them in real-time.

```bash
python ui.py
```
*(Navigate to `http://127.0.0.1:7860` in your browser to use the app).*

---

## How to use 💡

This project serves as a foundational example of building a neural network from scratch. It's ideal for:

*   **Learning Deep Learning Fundamentals:** Understand how neural networks work under the hood without the abstraction of libraries like TensorFlow or PyTorch.
*   **Vectorization & State Management:** See how standard Python `for` loops are replaced with lightning-fast C++ matrix operations via NumPy, using dictionaries to track forward caches and backward gradients.
*   **Experimentation:** Modify the `layer_dims` array in `train.py` to add or remove layers and observe how depth impacts accuracy.

---

## Project Structure 📂

```text
RawNET-MNIST/
├── .python-version
├── LICENSE
├── README.md
├── activations.py      # Pure math functions: ReLU, Softmax, and their derivatives
├── dataset.py          # MNIST fetcher, data flattener, and One-Hot encoder
├── engine.py           # The calculus core: dynamic initialization, forward pass, chain rule backprop
├── train.py            # Orchestrator: batch generator, training loop, and metrics logging
├── ui.py               # Gradio-powered interactive web canvas for live inference
├── trained_parameters.npy # Serialized weight matrices generated post-training
├── pyproject.toml
├── review.txt
├── src/
│   └── rawnet_mnist/
│       ├── __init__.py
└── uv.lock
```

---

## Contributing 🤝

Contributions are welcome! Whether it's improving documentation, adding features, or fixing bugs, your input is valuable.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to the project's style and includes tests if applicable.

---

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Important Links 🔗

*   **Repository:** [https://github.com/SouLCodes-main/RawNET-MNIST](https://github.com/SouLCodes-main/RawNET-MNIST)
*   **Author:** [SouLCodes-main](https://github.com/SouLCodes-main)

---

## Footer 👣

**RawNET-MNIST** | Built with ❤️ by [SouLCodes-main](https://github.com/SouLCodes-main)

[▲ Back to Top](#rawnet-mnist-)

---
