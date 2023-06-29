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
        sample_losses = self.forward(outputs, y)
        data_loss = np.mean(sample_losses)
        return data_loss

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


class Accuracy:
    def calculate(self, outputs, y):
        predictions = np.argmax(outputs, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy


# Initialize Dataset
X, y = vertical_data(samples=100, classes=3)

# Create Layers, Activations, Loss Function, and Accuracy Objects
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
loss_functions = Loss_CategoricalCrossEntropy()
accuracy_functions = Accuracy()

# Initialize Helper Variables
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
lowest_loss = 9999999

# Perform Iteration
for iteration in range(100000):
    # Generate new set of weights and biases
    dense1.weights += 0.05 * np.random.rand(2, 3)
    dense1.biases += 0.05 * np.random.rand(1, 3)
    dense2.weights += 0.05 * np.random.rand(3, 3)
    dense2.biases += 0.05 * np.random.rand(1, 3)

    # Implement 1st Dense Layer Forward Pass
    dense1.forward(X)

    # Implement Rectified Linear Activation Forward Pass
    activation1.forward(dense1.outputs)

    # Implement 2nd Dense Layer Forward Pass

    # Implement Softmax Activation Forward Pass

    # Calculate Loss

    # Calculate Accuracy

    # Check if current loss is less than lowest loss
    # Print the iteration number, loss, and accuracy

    # Copy the current iteration's weights, biases, and loss to helper variables

    # Else
    # Copy best weights and biases to dense object properties
