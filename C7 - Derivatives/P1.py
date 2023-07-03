# Import Libraries
import matplotlib.pyplot as plt
import numpy as np

# Method for Expression


def f(x):
    return (5*x**5) + (4*x**3) - 5


# Create X and Y coordinates
x = np.arange(0, 5, 0.0001)
y = f(x)

# Plot X and Y
plt.plot(x, y)

# Colors
colors = ['k', 'g', 'r', 'b', 'c']

# Method for Derivative


def analytical_derivative(x):
    return (25*x**4) + (12*x**2)

# Method for Tangent Line


def tangent_line(x, derivative, b):
    return (derivative * x) + b

# Iterate Range


# Show Plot
