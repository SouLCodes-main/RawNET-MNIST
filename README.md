# RawNET-MNIST 🧠

A pure-NumPy, functional-style neural network built entirely from scratch to classify handwritten digits. This project demonstrates the core calculus of deep learning by implementing a custom mathematics engine, He Initialization, ReLU/Softmax activations, and mini-batch gradient descent, all without relying on auto-differentiation frameworks like PyTorch or TensorFlow.

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![NumPy Version](https://img.shields.io/badge/NumPy-2.5.4-orange.svg)](https://numpy.org/)

---

## Table of Contents 📚

* [Project Title & Badges](#rawNET-MNIST-)
* [Description](#description)
* [Features](#features-)
* [Tech Stack](#tech-stack-)
* [Installation](#installation-)
* [Usage](#usage-)
* [Project Structure](#project-structure-)
* [Contributing](#contributing-)
* [License](#license-)
* [Important Links](#important-links-)
* [Footer](#footer-)

---

## Description 📝

RawNET-MNIST is an educational project that dives deep into the fundamental principles of neural networks. It's a complete implementation of a feedforward neural network using only NumPy, eschewing high-level deep learning libraries. The network is trained on the iconic MNIST dataset for handwritten digit classification. This project serves as an excellent resource for understanding the inner workings of neural networks, from matrix operations and activation functions to gradient descent and backpropagation.

---

## Features ✨

* **Pure NumPy Implementation:** No external deep learning frameworks (PyTorch, TensorFlow) used for core network operations.
* **Custom Mathematics Engine:** Implements all necessary mathematical operations from scratch.
* **He Initialization:** Utilizes He initialization for weights, suitable for ReLU activation functions.
* **Activation Functions:** Includes ReLU (Rectified Linear Unit) for hidden layers and Softmax for the output layer.
* **Mini-Batch Gradient Descent:** Implements efficient training using mini-batches.
* **MNIST Dataset Compatibility:** Specifically designed and trained on the MNIST handwritten digit dataset.
* **Functional Style:** Adopts a functional programming style for clarity and modularity.
* **Detailed Backpropagation:** Explicit implementation of the backpropagation algorithm.

---

## Tech Stack 💻

* **Language:** Python
* **Core Library:** NumPy
* **Data Loading:** Keras Datasets (via `keras` package)
* **Plotting:** Matplotlib
* **Build System:** uv_build

---

## Installation 🛠️

This project is designed to be run with Python 3.13+. The `pyproject.toml` file lists the necessary dependencies. You can install them using `pip` or `uv`.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SouLCodes-main/RawNET-MNIST.git
    cd RawNET-MNIST
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.

    Using `pip`:
    ```bash
    pip install -r requirements.txt # If a requirements.txt is generated or preferred
    # Or directly from pyproject.toml if your pip version supports it well:
    pip install --no-deps --editable . 
    ```
    (Note: The `pyproject.toml` has dependencies listed, a direct `pip install .` should work, but the provided `uv.lock` and `pyproject.toml` suggest a build system like `uv_build`. A simpler approach for direct execution is installing listed dependencies):

    ```bash
    pip install keras matplotlib numpy
    ```

    Using `uv` (if you have it installed):
    ```bash
      uv pip install . 
    ```

---

## Usage 🚀

The primary script for training and evaluating the model is `train.py`. Running this script will:

1.  Load and preprocess the MNIST dataset.
2.  Initialize the neural network parameters.
3.  Train the network using mini-batch gradient descent.
4.  Evaluate the trained model on the unseen test set.
5.  Display a plot of the learning curve (loss over epochs).

**To run the training process:**

```bash
python train.py
```

This will output the training progress, final test accuracy, and show a plot of the loss history.

---

## How to use 💡

This project serves as a foundational example of building a neural network from scratch. It's ideal for:

*   **Learning Deep Learning Fundamentals:** Understand how neural networks work under the hood without the abstraction of libraries like TensorFlow or PyTorch.
*   **Educational Purposes:** A practical demonstration for students and developers wanting to grasp concepts like forward propagation, backpropagation, gradient descent, and activation functions.
*   **Experimentation:** Modify hyperparameters (learning rate, epochs, batch size, hidden units) to observe their impact on training and accuracy.

**Example Use Case:**

After cloning and installing dependencies, simply execute `train.py` to see the network learn to classify handwritten digits from the MNIST dataset. The script will guide you through the process, showing the accuracy achieved on test data.

---

## Project Structure 📂

```
RawNET-MNIST/
├── .python-version
├── LICENSE
├── README.md
├── activations.py
├── dataset.py
├── engine.py
├── pyproject.toml
├── review.txt
├── src/
│   └── rawnet_mnist/
│       ├── __init__.py
├── train.py
└── uv.lock
```

*   `README.md`: This file.
*   `pyproject.toml`: Project metadata and build configuration.
*   `LICENSE`: The project's license file.
*   `activations.py`: Contains activation functions (ReLU, Softmax) and helper functions for predictions and accuracy.
*   `dataset.py`: Handles loading, preprocessing, and one-hot encoding of the MNIST dataset.
*   `engine.py`: Implements the core neural network logic: parameter initialization, forward propagation, cost computation, and backward propagation.
*   `train.py`: The main script for training the model, handling mini-batches, training loops, evaluation, and plotting.
*   `src/rawnet_mnist/`: Contains the main package structure, with `__init__.py` for the entry point.
*   `.python-version`, `review.txt`, `uv.lock`: Auxiliary files.

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

[▲ Back to Top](#rawNET-MNIST-)

---

---
**<p align="center">Generated by [ReadmeCodeGen](https://www.readmecodegen.com/)</p>**
