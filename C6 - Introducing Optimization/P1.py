# Import Libraries
import numpy as np
import nnfs

from nnfs.datasets import vertical_data

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
# Rectified Linear Unit (ReLu) Activation Class


class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

# Softmax Activation Class


# Loss Functions
# Loss Class


# Categorical Cross-Entropy Class


# Accuracy Class


# Initialize Dataset


# Create Layers, Activations, Loss Function, and Accuracy Objects


# Initialize Helper Variables


# Perform Iteration

# Generate new set of weights and biases

# Implement 1st Dense Layer Forward Pass

# Implement Rectified Linear Activation Forward Pass

# Implement 2nd Dense Layer Forward Pass

# Implement Softmax Activation Forward Pass

# Calculate Loss

# Calculate Accuracy

# Check if current loss is less than lowest loss
# Print the iteration number, loss, and accuracy

# Copy the current iteration's weights, biases, and loss to helper variables

# Else
# Copy best weights and biases to dense object properties
