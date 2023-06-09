inputs = [
    1.0, 2.0, 3.0, 2.5
]

weights1 = [
    0.2, 0.8, -0.5, 1.0
]

weights2 = [
    0.5, -0.91, 0.26, -0.5
]

weights3 = [
    -0.26, -0.27, 0.17, 0.87
]

biases = [2, 3, 0.5]

# output = [
#     # Neuron 1
#     inputs[0] * weights1[0] +
#     inputs[1] * weights1[1] +
#     inputs[2] * weights1[2] +
#     inputs[3] * weights1[3] + biases[0],

#     # Neuron 2
#     inputs[0] * weights2[0] +
#     inputs[1] * weights2[1] +
#     inputs[2] * weights2[2] +
#     inputs[3] * weights2[3] + biases[1],

#     # Neuron 3
#     inputs[0] * weights3[0] +
#     inputs[1] * weights3[1] +
#     inputs[2] * weights3[2] +
#     inputs[3] * weights3[3] + biases[2],
# ]

output = []

# Neuron 1
output.append(sum(input_n * weight for input_n,
              weight in zip(inputs, weights1)) + biases[0])

# Neuron 2
output.append(sum(input_n * weight for input_n,
              weight in zip(inputs, weights2)) + biases[1])

# Neuron 3
output.append(sum(input_n * weight for input_n,
              weight in zip(inputs, weights3)) + biases[2])

print(output)
