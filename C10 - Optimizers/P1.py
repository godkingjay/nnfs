# Import Libraries
import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()

# Layer Dense
# Dense Class


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

# Activation
# Rectified Linear Unit (ReLU) Activation Class


# Softmax Activation Class


# Loss
# Loss Class


# Categorical Cross-Entropy Loss Class


# Loss and Activation
# Categorical Cross-Entropy Loss and Softmax Activation Class


# Optimizer
# Stochastic Gradient Descent (SGD) Optimizer Class


# Accuracy Class


# Initialize Dataset


# Create 1st Dense Layer with 2 input features and 64 output values


# Create ReLU Activation (to be used with Dense Layer):


# Create 2nd Dense Layer (64 input features from prev layer and 3 output values)


# Create Softmax classifier's combined loss and activation


# Create Optimizer


# Train in loop
