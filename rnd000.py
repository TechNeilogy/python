import numpy as np

# x0 = np.array([0, 1, 2, 3, 4])
# y0 = np.array([2, 4, 6, 8, 0])

x0 = np.random.uniform(0, 1, 2000)
y0 = np.random.uniform(0, 1, 2000)

xd = np.subtract.outer(x0, x0)
yd = np.subtract.outer(y0, y0)

xd = xd * xd
yd = yd * yd

d2 = xd + yd

# print(d2)