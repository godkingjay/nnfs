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

        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# Loss
# Loss Class


class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        data_loss = np.mean(sample_losses)
        return data_loss

# Categorical Cross-Entropy Loss Class


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
            y_true = np.eye(labels)(y_true)

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

# Loss and Activation Functions
# Categorical Cross-Entropy Loss and Softmax Activation Class


class Loss_CategoricalCrossEntropy_Activation_Softmax():
    def __init__(self) -> None:
        self.activation = Activation_Softmax()
        self.loss_function = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs
        return self.loss_function.calculate(inputs, y_true)

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

# Optimizers
# Stochastic Gradient Descent (SGD) Optimizer Class


class Optimizer_SGD:
    def __init__(self, learning_rate=0., decay=0., momentum=0.) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer: Layer_Dense):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = (self.momentum * layer.weight_momentums) - \
                (self.current_learning_rate * layer.dweights)
            layer.weight_momentums = weight_updates

            bias_updates = (self.momentum * layer.bias_momentums) - \
                (self.current_learning_rate * layer.dbiases)
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

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

# Create 1st Dense Hidden Layer
dense1 = Layer_Dense(2, 64)

# Create 2nd Dense Hidden Layer
dense2 = Layer_Dense(64, 3)

# Create ReLU Activation Object
activation1 = Activation_ReLU()

# Create Softmax Classifier's combined Loss and Activation
loss_activation = Loss_CategoricalCrossEntropy_Activation_Softmax()

# Create SGD Optimizer
optimizer_sgd = Optimizer_SGD(decay=1e-3, momentum=0.9)

# Create Accuracy Object
accuracy_function = Accuracy()

# Epoch Iteration
for epoch in range(10001):
    # Implement 1st Dense Layer Forward Pass
    dense1.forward(X)

    # Implement ReLU Activation Forward Pass
    activation1.forward(dense1.outputs)

    # Implement 2nd Dense Layer Forward Pass
    dense2.forward(activation1.outputs)

    # Implement Categorical Cross-Entropy Loss and Softmax Activation Forward Pass
    loss = loss_activation.forward(dense2.outputs, y)

    # Calculate Accuracy from output of loss_activation and targets
    accuracy = accuracy_function.calculate(loss_activation.outputs, y)

    # Display Epoch, Loss and Accuracy
    if not epoch % 100 == 0:
        print(
            f'Epoch: {epoch},',
            f'Loss: {loss:.3f},',
            f'Accuracy: {accuracy:.3f}',
            f'Learning Rate: {optimizer_sgd.current_learning_rate}'
        )

    # Implement Categorical Cross-Entropy Loss and Softmax Activation Backward Pass
    loss_activation.backward(loss_activation.outputs, y)

    # Implement 2nd Dense Layer Backward Pass
    dense2.backward(loss_activation.dinputs)

    # Implement ReLu Activation Backward Pass
    activation1.backward(dense2.dinputs)

    # Implement 1st Dense Layer Backward Pass
    dense1.backward(activation1.dinputs)

    # Implement Optimizer pre-update parameters

    # Implement Optimizer update parameters

    # Implement Optimizer post-update parameters
    pass
