import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as py
from matplotlib import animation
import math

count = 8000
count1 = count - 1

size = 1000

x = np.random.normal(size / 2, size, count)
y = np.random.normal(size / 2, size, count)

# x = np.random.uniform(0, size, count)
# y = np.random.uniform(0, size, count)

a = np.random.normal(1, 0.7, count)

amax = a.max()
am = (a / amax) / 800 + 0.999

pa = np.random.rand(count) * np.pi * 2
pd = np.random.rand(count) * size / 2

for p in range(0, count):
    x[p] = math.sin(pa[p]) * pd[p] + size / 2
    y[p] = math.cos(pa[p]) * pd[p] + size / 2
#
# for p in range(count // 4, count // 2):
#     x[p] = math.sin(pa[p]) * pd[p] + size * 0.25
#     y[p] = math.cos(pa[p]) * pd[p] + size * 0.25
#
# for p in range(count // 2, count):
#     x[p] = math.sin(pa[p]) * pd[p] + size * 0.75
#     y[p] = math.cos(pa[p]) * pd[p] + size * 0.25

# for p in range(count // 2, count):
#     x[p] = math.sin(pa[p]) * pd[p] + size - size / 4
#     y[p] = math.cos(pa[p]) * pd[p] + size - size / 4

xm = np.zeros((count))
ym = np.zeros((count))

xg = np.zeros((count))
yg = np.zeros((count))

fig = py.figure(1)
ax = py.axes(xlim = (0, size), ylim = (0, size))
scat = ax.scatter([], [], s = 1, c = (0.2, 0.2, 0.5, 0.3))

def center_of_mass():
    xc = np.sum(x * a)
    yc = np.sum(y * a)
    return xc / count, yc / count

# def pulla(x0, y0, xc, yc):
#
#     xd = x0 - xc
#     yd = y0 - yc
#
#     d2 = xd * xd + yd * yd
#     d2 = 1 / np.sqrt(d2) + 0.00001
#     # cd = 1 / (math.sqrt(xd * xd + yd * yd) + 0.00001)
#
#     sf = 0.005
#     xd = xd * d2 * sf
#     yd = yd * d2 * sf
#
#     return xd, yd

def pull(xc, yc):

    global x, y, xm, ym, xg, yg

    xd = - (x - xc)
    yd = - (y - yc)

    d2 = xd * xd + yd * yd
    d2 = 1 / np.sqrt(d2) + 0.00001

    sf = 0.045
    xp = xd * d2 * sf
    yp = yd * d2 * sf

    xm = xm * am + xp
    ym = ym * am + yp

    x = x + xm
    y = y + ym

    xg = x - xc + size / 2
    yg = y - yc + size / 2

    for p in range(0, count):
        xd[p] = min(size, max(0, xd[p]))
        yd[p] = min(size, max(0, yd[p]))

def init():
    scat.set_offsets([])
    return scat,

def animate(i):

    xc, yc = center_of_mass()

    pull(xc, yc)

    # for p in range(0, count):
    #
    #     #xp, yp, zp = pull(xc, yc, zc, x[p], y[p], z[p])
    #     xp, yp = pull(xc, yc, x[p], y[p])
    #
    #     xm[p] = xm[p] * m + xp
    #     ym[p] = ym[p] * m + yp
    #     #zm[p] = zm[p] * m + zp
    #
    #     x[p] = min(size, max(0, x[p] + xm[p]))
    #     y[p] = min(size, max(0, y[p] + ym[p]))
    #
    #     # x[p] = x[p] + xm[p]
    #     # y[p] = y[p] + ym[p]
    #     #z[p] = min(size, max(0, z[p] + zm[p]))

    data = np.hstack((xg[:count1, np.newaxis], yg[:count1, np.newaxis]))
    scat.set_offsets(data)
    # if i % 10 == 0:
    #     scat.set_sizes(z / size * 20)

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