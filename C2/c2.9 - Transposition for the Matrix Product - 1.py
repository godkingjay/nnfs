import numpy as np

a = [1, 2, 3]

# 1D array
print(np.array(a))

# 2D array
# print(np.array([a]))
print(np.expand_dims(np.array(a), axis=0))
