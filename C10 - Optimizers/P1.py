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


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# Softmax Activation Class


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_outputs, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            single_outputs = single_outputs.reshape(-1, 1)
            jacobian_matrix = np.diagflat(
                single_outputs) - np.dot(single_outputs, single_outputs.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# Loss
# Loss Class


class Loss:
    def calculate(self, y_pred, y_true):
        samples_losses = self.forward(y_pred, y_true)
        data_loss = np.mean(samples_losses)
        return data_loss

# Categorical Cross-Entropy Loss Class


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_true, 1e-7, 1 - 1e-7)

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

# Loss and Activation
# Categorical Cross-Entropy Loss and Softmax Activation Class


class Loss_CategoricalCrossEntropy_Activation_Softmax():
    def __init__(self) -> None:
        self.activation = Activation_Softmax()
        self.loss_function = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs
        return self.loss_function.calculate(self.outputs, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[
            range(samples),
            y_true
        ] -= 1
        self.dinputs = self.dinputs / samples

# Optimizer
# Stochastic Gradient Descent (SGD) Optimizer Class


class Optimizer_SGD:
    def __init__(self, learning_rate=1.0) -> None:
        self.learning_rate = learning_rate

    def update_params(self, layer: Layer_Dense):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

# Accuracy Class


class Accuracy:
    def calculate(self, y_pred, y_true):
        predictions = np.argmax(y_pred, axis=1)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        accuracy = np.mean(predictions == y_true)
        return accuracy


# Initialize Dataset
X, y = spiral_data(samples=100, classes=3)

# Create 1st Dense Layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)

# Create ReLU Activation (to be used with Dense Layer):
activation = Activation_ReLU()

# Create 2nd Dense Layer (64 input features from prev layer and 3 output values)


# Create Softmax classifier's combined loss and activation


# Create Optimizer


# Train in loop
