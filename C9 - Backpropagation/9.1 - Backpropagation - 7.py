import numpy as np

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

z = np.dot(x, w) + b

y = max(z, 0)

dvalue = 1.0

drelu_dz = dvalue * (1. if z > 0 else 0.)
print(drelu_dz)

dsum_dxw0 = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
# print(drelu_dxw0)

dsum_dxw1 = 1
drelu_dxw1 = drelu_dz * dsum_dxw1
# print(drelu_dxw1)

dsum_dxw2 = 1
drelu_dxw2 = drelu_dz * dsum_dxw2
# print(drelu_dxw2)

dsum_db = 1
drelu_db = drelu_dz * dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

dmul_dx0 = w[0]
drelu_dx0 = drelu_dxw0 * dmul_dx0
print(drelu_dx0)