# Import Libraries
import matplotlib.pyplot as plt
import numpy as np

# Method for Expression


def f(x):
    return (3*x**2) + (5*x)


# Create X and Y coordinates
x = np.arange(0, 5, 0.0001)
y = f(x)

# Plot X and Y
plt.plot(x, y)

# Colors
colors = ['k', 'g', 'r', 'b', 'c']

# Method for Derivative


def analytical_derivative(x):
    return (6*x) + 5

# Method for Tangent Line


def tangent_line(x, derivative, b):
    return (derivative * x) + b


# Iterate Range
for i in range(5):
    # Create Delta
    p2_delta = 0.0001

    # X1 and X2
    x1 = i
    x2 = x1 + p2_delta

    # Y1 and Y2
    y1 = f(x1)
    y2 = f(x2)

    # Get Derivative
    derivative = analytical_derivative(x1)

    # Get y-intercept
    b = y2 - (derivative * x2)

    # Create X and Y coordinates for tangent line
    xd = [x1 - 0.9, x1, x1 + 0.9]
    yd = [tangent_line(point, derivative, b) for point in xd]

    # Plot tangent line
    plt.scatter(x1, y1, c=colors[i])
    plt.plot(xd, yd, c=colors[i])

    print('Analytical derivative for f(x)',
          f'where x = {x1} is {derivative}')

plt.show()
