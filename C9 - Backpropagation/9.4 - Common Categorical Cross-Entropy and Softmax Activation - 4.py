import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()


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


class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.dinputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
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


class Loss:
    def calculate(self, outputs, y_true):
        sample_losses = self.forward(outputs, y_true)
        data_loss = np.mean(sample_losses)
        return data_loss


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

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossEntropy():
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


class Accuracy:
    def calculate(self, outputs, y_true):
        predictions = np.argmax(outputs, axis=1)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        accuracy = np.mean(predictions == y_true)
        return accuracy


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
dense2 = Layer_Dense(3, 3)

activation1 = Activation_ReLU()
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
accuracy_function = Accuracy()

# Perform Forward Passes
dense1.forward(X)
activation1.forward(dense1.outputs)
dense2.forward(activation1.outputs)
loss = loss_activation.forward(dense2.outputs, y)
accuracy = accuracy_function.calculate(loss_activation.outputs, y)

print(loss_activation.outputs[:5])
print('Loss:', loss)

print("Accuracy:", accuracy)

print()

loss_activation.backward(loss_activation.outputs, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.weights)
print(dense1.dweights)

print()

print(dense1.biases)
print(dense1.dbiases)

print()

print(dense2.weights)
print(dense2.dweights)

print()

print(dense2.biases)
print(dense2.dbiases)
