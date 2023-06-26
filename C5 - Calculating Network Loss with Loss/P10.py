# Import Libraries
import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()

# Layers
# Dense Class


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

# Activation Functions
# Rectified Linear Unit (ReLU) Activation Class


class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

# Softmax Activation Class


# Loss Function
# Loss Class


# Categorical Cross-Entropy Loss Class


# Accuracy Class


# Initialize Dataset


# Create 1st Hidden Layer


# Implement Rectified Linear Unit (ReLU) Activation


# Create 2nd Hidden Layer


# Implement Softmax Activation


# Calculate Loss


# Calculate Accuracy
