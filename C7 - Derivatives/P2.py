# Import Libraries
import matplotlib.pyplot as plt
import numpy as np

# Method for Expression


def f(x):
    return (x**3) + (2*x**2) - (5*x) + 7


# Create x and y coordinates
x = np.arange(0, 5, 0.0001)
y = f(x)

# Plot x and y coordinates
plt.plot(x, y)

# Colors
colors = ['k', 'g', 'r', 'b', 'c']

# Method for Derivative


def analytical_derivative(x):
    return (3*x**2) + (4*x) - 5

# Method for tangent line


def tangent_line(x, derivative, b):
    return (derivative * x) + b


# Iterate Range
for i in range(5):
    # Create Delta
    p2_delta = 0.0001

    # x coordinate for two points

    # y coordinate for two points

    # Get Derivative

    # Get y-intercept

    # Create x and y coordinates for tangent line

    # Plot tangent line

    # Print Derivative

# Show Plot
plt.show()
