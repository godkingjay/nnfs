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


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities

# Loss Functions
# Loss Class


class Loss:
    def calculate(self, outputs, y):
        pass

# Categorical Cross-Entropy Class


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        neg_log = -np.log(correct_confidences)
        return neg_log

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
