inputs = [
    1.0, 2.0, 3.0, 2.5
]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
]

biases = [2, 3, 0.5]

layer_outputs = []

# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += weight * n_input
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)

# Get the output for each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Neuron output
    neuron_output = sum(weight * n_input for weight,
                        n_input in zip(neuron_weights, inputs)) + neuron_bias
    # Append to layer outputs
    layer_outputs.append(neuron_output)

print(layer_outputs)
