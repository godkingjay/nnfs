# Import Packages
import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()

# Layer
# Dense Class


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

# Activations
# Rectified Linear Unit (ReLU) Class


class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

# Softmax Class


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities

# Loss
# Loss Class


# class Loss:
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
    def forward(self, outputs, y):
        predictions = np.argmax(outputs, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy


# Initialize Dataset
X, y = spiral_data(samples=100, classes=3)

# Implement Dense
dense1 = Layer_Dense(2, 3)
dense1.forward(X)

# Implement ReLU
activation1 = Activation_ReLU()
activation1.forward(dense1.outputs)

# Implement Dense 2
dense2 = Layer_Dense(3, 3)
dense2.forward(activation1.outputs)

# Implement Softmax
activation2 = Activation_Softmax()
activation2.forward(dense2.outputs)

# Implement Loss
loss1 = Loss_CategoricalCrossEntropy()
loss_value = loss1.calculate(activation2.outputs, y)

print("Loss: ", loss_value)

# Implement Accuracy
accuracy1 = Accuracy()
accuracy_value = accuracy1.forward(activation2.outputs, y)

print("Accuracy: ", accuracy_value)