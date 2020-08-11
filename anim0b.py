import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as py
from matplotlib import animation

count = 1000
count1 = count - 1

size = 1000
dSize = 20

x = np.random.normal(size / 2, 1, count)
y = np.random.normal(size / 2, 1, count)
d = np.random.normal(dSize / 2, 1, count)

# py.figure(1)
# py.scatter(x, y, s=60)
# py.axis([0, 1, 0, 1])
# py.show()

fig = py.figure(1)
ax = py.axes(xlim = (0, size), ylim = (0, size))
scat = ax.scatter([], [], s = dSize / 2, c = (0.2, 0.2, 0.5, 0.3))

def init():
    scat.set_offsets([])
    return scat,

def animatex(i):
    scat.set_offsets([x[:i], y[:i]])
    return scat,

def animate(i):
    for j in range(0, count):
        zz = (abs(x[j] - size/2)) / size * 8 + 0.25
        x[j] = min(size, max(0, x[j] + np.random.normal(0, zz)))
        y[j] = min(size, max(0, y[j] + np.random.normal(0, zz)))
        d[j] = min(dSize, max(1, d[j] + np.random.normal(0, zz)))
    # j0 = np.random.randint(0, count)
    # x[j0] = size / 2
    # y[j0] = size / 2
    # d[j0] = dSize / 2
    data = np.hstack((x[:count1, np.newaxis], y[:count1, np.newaxis]))
    scat.set_offsets(data)
    scat.set_sizes(d)
    return scat,

anim = animation.FuncAnimation(
    fig,
    animate,
    init_func = init,
    frames = len(x),
    interval = 5,
    blit = True,
    repeat = True
)

plt.show()