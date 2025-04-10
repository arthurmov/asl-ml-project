# asl-ml-project
### Simple Neural Network & Autoencoder Exploration

This project explores neural networks through two parallel approaches:

1. A fully manual neural network implemented with NumPy
2. A PyTorch-based autoencoder trained on gesture data

The project demonstrates core machine learning concepts like forward and backward propagation, training loops, and representation learning.

Developed as part of a CISC 3440 Machine Learning course.

---

### Project Overview

### Part 1: Manual Neural Network (NumPy)
- Implements a feedforward neural network from scratch.
- Trains on a small toy dataset like `[[1, 1], [1, 0], [0, 1], [0, 0]]`.
- Performs forward and backward propagation manually.
- Optimizes using stochastic gradient descent.
- Visualizes training error over epochs using `matplotlib`.

### Part 2: Autoencoder (PyTorch)
- Defines a custom `Autoencoder` class using `torch.nn.Module`.
- Trains the model to compress and reconstruct high-dimensional input data (e.g. hand poses).
- Uses `torch.utils.data.DataLoader` for batching.
- Optimizes with `Adam` and `MSELoss`.
- Visualizes embeddings and reconstruction performance.
