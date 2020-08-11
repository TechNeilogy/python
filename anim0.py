import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as py
from matplotlib import animation
import math

epsilon = 0.00001

starCount = 8000
width = 1.0

starCount_1 = starCount - 1
width_2 = width / 2

x = np.zeros(starCount)
y = np.zeros(starCount)
# z = np.zeros(starCount)

mass = np.random.uniform(0.1, 1.0, starCount)

massMax = mass.max()
massMean = mass.mean()

polarAngleXY = np.random.rand(starCount) * np.pi * 2
# polarAngleXZ = np.random.rand(starCount) * np.pi * 2
polarDistance = np.random.rand(starCount) * width_2

for p in range(0, starCount):
    x[p] = math.sin(polarAngleXY[p]) * polarDistance[p] + width_2
    y[p] = math.cos(polarAngleXY[p]) * polarDistance[p] + width_2
    # z[p] = math.cos(polarAngleXZ[p]) * polarDistance[p] + width_2

x_momentum = np.zeros((starCount))
y_momentum = np.zeros((starCount))
# zMomentum = np.zeros((starCount))

fig = py.figure(1)

ax = py.axes(
    xlim = (0, width),
    ylim = (0, width)
)

scat = ax.scatter(
    [],
    [],
    s = 1,
    c = (0.2, 0.2, 0.5, 0.3)
)

def center_of_mass():
    xc = np.sum(x * mass)
    yc = np.sum(y * mass)
    return xc / starCount, yc / starCount

def pull(xc, yc):

    global x, y, x_momentum, y_momentum

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

    xMassCenter = np.sum(x * mass) / starCount
    yMassCenter = np.sum(y * mass) / starCount

    xDelta = x - xMassCenter
    yDelta = y - yMassCenter

    distance = np.sqrt(
        xDelta * xDelta +
        yDelta * yDelta
    )

    distance2 = distance * distance + epsilon

    attraction = massMean / distance2

    xg = x + xDelta * attraction
    yg = y + yDelta * attraction

    data = np.hstack((
        xg[:starCount_1, np.newaxis],
        yg[:starCount_1, np.newaxis]
    ))

    scat.set_offsets(data)

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