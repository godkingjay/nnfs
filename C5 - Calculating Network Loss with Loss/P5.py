# Import Packages
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

# Activations
# Rectified Linear Unit(ReLU) Class


class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

# Softmax Class


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Loss
# Loss Class


# Categorical Cross-Entropy Class


# Accuracy Class


# Initialization of Datasets


# Create 1st Hidden Layer


# Implement ReLU Activation


# Create 2nd Hidden Layer


# Implement Softmax Activation


# Calculate Loss


# Calculate Accuracy
