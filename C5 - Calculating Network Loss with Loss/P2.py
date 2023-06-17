# import libraries
import numpy as np
import nnfs

from nnfs.datasets import spiral_data

# Layer
# Dense Class
class Layer_Dense:
  def __init__(self, n_inputs, n_neurons) -> None:
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
    
  def forward(self, inputs):
    self.outputs = np.dot(input, self.weights) + self.biases

# Activations
# Rectified Linear Unit (ReLU) Class


#  Softmax Activation Class


# Loss
# Loss Class


# Categorical Cross-Entropy Loss Class (Loss)


# Accuracy Class


# Initialize dataset


# Apply Dense Layer


# Apply ReLU Activation


# Apply Softmax Activation


# Apply Loss Function


# Calculate Accuracy