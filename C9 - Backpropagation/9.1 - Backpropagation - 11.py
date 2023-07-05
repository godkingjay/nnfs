import numpy as np

dvalues = np.array([[1., 1., 1.]])

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

dinputs = np.dot(dvalues[0], weights.T)

print(dinputs)
