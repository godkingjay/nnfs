import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

exp_values = []

for output in layer_outputs:
    exp_values.append(np.exp(output))

print("Exponentiated Values:")
print(exp_values)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print("Sum of Exponentiated values:")
print(norm_base)

print("Normalized values:")
print(norm_values)

print("Sum of Normalized values:")
print(sum(norm_values))
