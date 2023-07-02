import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 2*x**2


x = np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x, y)

p2_delta = 0.0001
x1 = 2
x2 = x1 + p2_delta

y1 = f(x1)
y2 = f(x2)

print((x1, y1), (x2, y2))

approximate_derivative = (y2 - y1) / (x2 - x1)
b = y2 - approximate_derivative * x2


def tangent_line(x):
    return approximate_derivative * x + b


xd = [x1 - 0.9, x2, x1 + 0.9]
yd = [tangent_line(i) for i in xd]
plt.plot(xd, yd)

print(xd, yd)

print('Approximate derivative for f(x)',
      f'where x = {x1} is {approximate_derivative}')

plt.show()
