# Import Libraries
import numpy as np
import nnfs

from nnfs.datasets import vertical_data

nnfs.init()

# Layers
# Dense Layer Class


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

# Activation Functions
# Rectified Linear Unit (ReLU) Activation Class


# Softmax Activation Class


# Loss Functions
# Loss Class


# Categorical Cross-Entropy Loss Class


# Accuracy Class


# Initialize Dataset


# Create Hidden Dense Layers


# Create object for Activation Functions


# Create object for Loss and Accuracy


# Optimization
