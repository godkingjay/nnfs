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
for i in range(5):
    # Create Delta
    p2_delta = 0.0001

    # Create X variable for two points
    x1 = i
    x2 = x1 + p2_delta

    # Create Y variable for two points
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

    # Print Derivative
    print('Analytical derivative for f(x)',
          f'where x = {x1} is {derivative}')

# Show Plot
plt.show()
