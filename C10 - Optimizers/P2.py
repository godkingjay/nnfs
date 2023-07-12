# Import Libraries
import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()

# Layers
# Dense Layer Class


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# Activation Functions
# Rectified Linear Unit (ReLU) Activation Class
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

# Softmax Activation Class


# Loss
# Loss Class


# Categorical Cross-Entropy Loss Class


# Loss and Activation Functions
# Categorical Cross-Entropy Loss and Softmax Activation Class


# Optimizers
# Stochastic Gradient Descent (SGD) Optimizer Class


# Accuracy Class


# Initialize Dataset


# Create 1st Dense Hidden Layer


# Create 2nd Dense Hidden Layer


# Create ReLU Activation Object


# Create Softmax Classifier's combined Loss and Activation


# Create Loss and Accuracy Object


# Epoch Iteration
