# 🧠 MNIST Digit Classifier from Scratch using NumPy

This project implements a simple **fully connected neural network** to classify handwritten digits from the MNIST dataset — **completely from scratch**, using only **NumPy**. No PyTorch, no TensorFlow.

> Goal: Understand and implement the core mechanics of forward pass, backpropagation, and training from the ground up.



## Dataset

- **Dataset**: [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- **Source**: Loaded via `sklearn.datasets.fetch_openml()`
- **Shape**: 70,000 grayscale images (28x28), flattened into 784 features



## Model Architecture

| Layer        | Size          | Activation |
|--------------|---------------|------------|
| Input        | 784           | -          |
| Hidden Layer | 10 neurons    | ReLU       |
| Output       | 10 neurons    | Softmax    |

> This is a single hidden-layer fully connected network.



## Features

- Manual implementation of:
  - Weight initialization
  - Forward pass
  - ReLU and softmax
  - Cross-entropy loss
  - Backpropagation
  - Gradient descent
- One-hot encoding for labels
- Accuracy and loss tracking
- No ML libraries used (no PyTorch, TensorFlow, or Keras)


## Results

- Train Accuracy: ~90%
- Test Accuracy: ~89%
- Layers: 1 hidden layer
- Tools: `numpy`, `sklearn` for dataset only


## How to Run

1. Clone this repo
- git clone [MNIST-from-scratch](https://github.com/sejal-ml/MNIST-from-scratch.git)
- cd MNIST-from-scratch

2. Install dependencies
- pip install numpy scikit-learn matplotlib

3. Run the notebook
- jupyter notebook [MNIST.ipynb](https://github.com/sejal-ml/MNIST.ipynb)
