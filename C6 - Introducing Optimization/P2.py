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

# Create 1st and 2nd Hidden Layer
dense1 = Layer_Dense(2, 3)
dense2 = Layer_Dense(3, 3)

# Create Activation Functions objects
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()

# Create Loss and Accuracy Function objects
loss_function = Loss_CategoricalCrossEntropy()
accuracy_function = Accuracy()

# Initialize Helper Variables
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
lowest_loss = 9999999

# Perform Iteration
for iteration in range(10000):
    # Generate new set of weights and biases
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # Perform 1st Dense Hidden Layer Forward Pass
    dense1.forward(X)

    # Implement Rectified Linear Unit (ReLU) Activation Forward Pass
    activation1.forward(dense1.outputs)

    # Perform 2nd Dense Hidden Layer Forward Pass
    dense2.forward(activation1.outputs)

    # Implement Softmax Activation Forward Pass
    activation2.forward(dense2.outputs)

    # Calculate Loss
    loss = loss_function.calculate(activation2.outputs, y)

    # Calculate Accuracy
    accuracy = accuracy_function.calculate(activation2.outputs, y)

    # If Loss is Lower, Print and Save Weights and Biases
    if loss < lowest_loss:
        print(
            f'New set of weights found, iteration: {iteration}, loss: {loss}, accuracy: {accuracy}')
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    # Else Restore Weights and Biases
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
