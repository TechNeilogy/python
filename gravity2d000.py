import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as py
from matplotlib import animation
import math

epsilon = 0.00001

star_count = 2000
width = 1.0

width_2 = width / 2

x = np.zeros(star_count)
y = np.zeros(star_count)

fig = py.figure(1, figsize=(5, 5))
fig.patch.set_facecolor((0.1, 0.1, 0.1, 1))

ax = py.axes(xlim = (0, width), ylim = (0, width))
ax.set_facecolor((0.0, 0.0, 0.0, 1))

scat = ax.scatter(x, y, s = 1, c = (0.2, 0.2, 0.5, 0.3))

# mass = np.random.uniform(0.5, 1.0, star_count)
mass = np.random.normal(0.8, 0.1, star_count)

mass_max = mass.max()
mass_mean = mass.mean()
mass_fraction = mass / mass.sum()

x_momentum = np.zeros((star_count))
y_momentum = np.zeros((star_count))

def hot_spots():

    spot_count = 4

    hX = np.zeros(spot_count)
    hY = np.zeros(spot_count)

    h_polar_angle_xy = np.random.rand(spot_count) * np.pi * 2

    mn = 0.0
    mx = 0.75
    md = mx - mn

    h_polar_distance = (np.random.rand(spot_count) * md + mn) * width_2

    for p in range(0, spot_count):
        hX[p] = math.sin(h_polar_angle_xy[p]) * h_polar_distance[p] + width_2
        hY[p] = math.cos(h_polar_angle_xy[p]) * h_polar_distance[p] + width_2

    global x, y

    polar_angle_xy = np.random.rand(star_count) * np.pi * 2

    mn = 0.0
    mx = np.random.rand(star_count) / 5
    md = mx - mn

    polar_distance = (np.random.rand(star_count) * md + mn)

    for p in range(0, star_count):
        spot = np.random.randint(0, spot_count)
        x[p] = math.sin(polar_angle_xy[p]) * polar_distance[p] + hX[spot]
        y[p] = math.cos(polar_angle_xy[p]) * polar_distance[p] + hY[spot]

def rectangle(
        x_size,
        y_size
):
    global x, y

    x_size = x_size / 2
    y_size = y_size / 2

    x = np.random.uniform(0.5 - x_size, 0.5 + x_size, star_count)
    y = np.random.uniform(0.5 - y_size, 0.5 + y_size, star_count)

graph = ax.scatter(
    x,
    y,
    s = 1,
    c = (1, 1, 1, 0.3)
)

rectangle(0.1, 1)
# hot_spots()

massSums = np.add.outer(mass, mass)

def animate(frame):

    global graph
    global x, y
    global x_momentum, y_momentum
    global massSums

    xd = np.subtract.outer(x, x)
    yd = np.subtract.outer(y, y)

    distances = xd * xd + yd * yd + 0.000001

    attraction = massSums / distances * 0.000000005

    x_delta = np.sum(xd * attraction, 0)
    y_delta = np.sum(yd * attraction, 0)

    ambient_friction = 0.9999
    x_momentum = x_delta + x_momentum * ambient_friction
    y_momentum = y_delta + y_momentum * ambient_friction

    x += x_momentum
    y += y_momentum

    x_mass_center = np.sum(x * mass_fraction)
    y_mass_center = np.sum(y * mass_fraction)

    x = (x - x_mass_center) + width_2
    y = (y - y_mass_center) + width_2

    data = np.hstack((
        x[:star_count, np.newaxis],
        y[:star_count, np.newaxis]
    ))

    graph.set_offsets(data)

    return graph,

anim = animation.FuncAnimation(
    fig,
    animate,
    frames = len(x),
    interval = 5,
    blit = True,
    repeat = True
)

plt.show()