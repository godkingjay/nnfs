import numpy as np

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

z = np.dot(x, w) + b
y = max(z, 0)

dvalue = 1.0

drelu = dvalue * (1. if z > 0 else 0.)
print(drelu)
