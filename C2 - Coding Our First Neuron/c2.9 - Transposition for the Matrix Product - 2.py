import numpy as np

a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a])
b = np.array([b])

# Error as array shapes are (1, 3) and (1, 3)
# dot_output = np.dot(a, b)

# Transpose b to (3, 1)
b = np.transpose(b)

dot_output = np.dot(a, b)

print(dot_output)
