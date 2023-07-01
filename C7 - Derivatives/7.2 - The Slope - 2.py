def f(x):
    return 2*x**2


p2_delta = 0.0001

x1 = 1
x2 = x1 + p2_delta  # add delta

y1 = f(x1)
y2 = f(x2)

approximate_derivative = (y2 - y1) / (x2 - x1)
print(approximate_derivative)
