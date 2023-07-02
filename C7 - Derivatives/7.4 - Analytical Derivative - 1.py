import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 2*x**2


x = np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x, y)

colors = ['k', 'g', 'r', 'b', 'c']


def analytical_derivative(x):
    return 4*x


def tangent_line(x, derivative, b):
    return (derivative * x) + b


for i in range(5):
    p2_delta = 0.0001

    x1 = i
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)

    derivative = analytical_derivative(x1)
    b = y2 - (derivative * x2)

    xd = [x1 - 0.9, x1, x1 + 0.9]
    yd = [tangent_line(point, derivative, b) for point in xd]

    plt.scatter(x1, y1, c=colors[i])
    plt.plot(xd, yd, c=colors[i])

    print('Analytical derivative for f(x)',
          f'where x = {x1} is {derivative}')

plt.show()
