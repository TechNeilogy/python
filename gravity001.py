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

star_count_1 = star_count# - 1
width_2 = width / 2
width_4 = width / 4

x = np.zeros(star_count)
y = np.zeros(star_count)
z = np.zeros(star_count)

mass = np.random.uniform(1.0, 1.0, star_count)

# Note: Using colors really slows things down.
mass_colors = np.vstack((
    (mass),
    (1 - mass),
    (np.zeros(star_count)),
    (np.ones(star_count))
)).T

mass_max = mass.max()
mass_mean = mass.mean()
mass_fraction = mass / mass.sum()

x_momentum = np.zeros((star_count))
y_momentum = np.zeros((star_count))
z_momentum = np.zeros((star_count))

def hot_spots():

    spot_count = 100

    hX = np.zeros(spot_count)
    hY = np.zeros(spot_count)
    hZ = np.zeros(spot_count)

    h_polar_angle_xy = np.random.rand(spot_count) * np.pi * 2
    h_polar_angle_xz = np.random.rand(spot_count) * np.pi * 2

    mn = 0.0
    mx = 0.85
    md = mx - mn

    h_polar_distance = (np.random.rand(spot_count) * md + mn) * width_2

    for p in range(0, spot_count):
        hX[p] = math.sin(h_polar_angle_xy[p]) * h_polar_distance[p] + width_2
        hY[p] = math.cos(h_polar_angle_xy[p]) * h_polar_distance[p] + width_2
        hZ[p] = math.cos(h_polar_angle_xz[p]) * h_polar_distance[p] + width_2

    global x, y, z

    polar_angle_xy = np.random.rand(star_count) * np.pi * 2
    polar_angle_xz = np.random.rand(star_count) * np.pi * 2

    mn = 0.0
    mx = np.random.rand(star_count) / 20
    md = mx - mn

    polar_distance = (np.random.rand(star_count) * md + mn)

    for p in range(0, star_count):
        spot = np.random.randint(0, spot_count)
        x[p] = math.sin(polar_angle_xy[p]) * polar_distance[p] + hX[spot]
        y[p] = math.cos(polar_angle_xy[p]) * polar_distance[p] + hY[spot]
        z[p] = math.cos(polar_angle_xz[p]) * polar_distance[p] + hZ[spot]


# disc()
# hoedown()
# two()
hot_spots()
#two()
# uniform()
plt.rcParams['grid.color'] = (0.0, 0.3, 0.3, 1)

fig = plt.figure(figsize=(4, 4))

fig.patch.set_facecolor((0.0, 0.0, 0.0, 1))
ax = fig.add_subplot(
    111,
    projection="3d"
)
ax.set_position((0,0,1,1))
ax.set_facecolor((0.0, 0.0, 0.0, 1))

graph = ax.scatter(
    x,
    y,
    z,
    s = 1,
    c = (0.9, 0.9, 0.9, 0.3)
)
#text = fig.text(0, 1, "TEXT", va='top')  # for debugging

# fig = py.figure(1)
#
# from mpl_toolkits.mplot3d import Axes3D
#
# ax = fig.add_subplot(111, projection='3d')


# ax = py.axes(
#     xlim = (0, width),
#     ylim = (0, width)
# )

# scale = width_2 * 0.4
# ax = py.axes(
#     xlim = (width_2 - scale, width + scale),
#     ylim = (width_2 - scale, width + scale)
# )

# scat = ax.scatter(
#     x,
#     y,
#     z,
#     s = 1, #mass * 5,
#     c = [1, 1, 1, 0.5],
#     # c = mass_colors
#     animated=True
# )

# fig.set_facecolor((0,0,0,1))
# ax.set_facecolor((0,0,0,1))

# def init():
#     # scat.set_offsets([x, y, z])
#     return scat,

massX = np.add.outer(mass, mass)

def animate(frame):

    global graph
    global x, y, z
    global x_momentum, y_momentum, z_momentum
    global massX

    #for q in range(0, 4):

    xd = np.subtract.outer(x, x)
    yd = np.subtract.outer(y, y)
    zd = np.subtract.outer(z, z)

    d = xd * xd + yd * yd + zd * zd + 0.000001

    attraction = massX / d * 0.000000005

    x_delta = np.sum(xd * attraction, 0)
    y_delta = np.sum(yd * attraction, 0)
    z_delta = np.sum(zd * attraction, 0)

    ambient_friction = 0.9999
    x_momentum = x_delta + x_momentum * ambient_friction
    y_momentum = y_delta + y_momentum * ambient_friction
    z_momentum = z_delta + z_momentum * ambient_friction

    x += x_momentum
    y += y_momentum
    z += z_momentum

    # print(x[0])

    # x_mass_center = np.sum(x * mass_fraction)
    # y_mass_center = np.sum(y * mass_fraction)
    # z_mass_center = np.sum(z * mass_fraction)

    # x = (x - x_mass_center) + width_2
    # y = (y - y_mass_center) + width_2
    # z = (z - z_mass_center) + width_2
    #
    # data = np.hstack((
    #     x[:star_count_1, np.newaxis],
    #     y[:star_count_1, np.newaxis],
    #     z[:star_count_1, np.newaxis]
    # ))

    # x += 0.1
    graph._offsets3d = (x, y, z)

    return graph,

ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

anim = animation.FuncAnimation(
    fig,
    animate,
    # init_func = init,
    frames = len(x),
    interval = 1,
    blit = False,
    repeat = True
)

plt.show()





# scat._offsets3d = (
    #     x,
    #     y,
    #     z
    # )

    # data = [[x, y, z]]

    # scale = 2
    # xs = (x - x_mass_center) * scale + width_2
    # ys = (y - y_mass_center) * scale + width_2
    # data = np.hstack((
    #     xs[:starCount_1, np.newaxis],
    #     ys[:starCount_1, np.newaxis]
    # ))

    #scat.set_offsets(data)








def animate2(i):

    global x, y, x_momentum, y_momentum

    x_mass_center = np.sum(x * mass_fraction)
    y_mass_center = np.sum(y * mass_fraction)

    x_delta = x - x_mass_center
    y_delta = y - y_mass_center

    distance = np.sqrt(
        x_delta * x_delta +
        y_delta * y_delta
    )

    distance_squared = distance * distance + epsilon

    attraction_scale_factor = 0.00001
    # attraction = (mass * 0.1 + mass_mean) / distance_squared * attraction_scale_factor
    attraction = mass_mean / distance_squared * attraction_scale_factor

    ambient_friction = 1 # 0.999999
    x_momentum = x_delta * attraction + x_momentum * ambient_friction
    y_momentum = y_delta * attraction + y_momentum * ambient_friction

    x = x - x_momentum
    y = y - y_momentum

    # x = x - x_mass_center + width_2
    # y = y - y_mass_center + width_2
    #
    # data = np.hstack((
    #     x[:starCount_1, np.newaxis],
    #     y[:starCount_1, np.newaxis]
    # ))

    scale = 10000
    xs = (x - x_mass_center) * scale + width_2
    ys = (y - y_mass_center) * scale + width_2
    data = np.hstack((
        xs[:star_count_1, np.newaxis],
        ys[:star_count_1, np.newaxis]
    ))

    scat.set_offsets(data)

    return scat,

def animate3(frame):

    global x, y, x_momentum, y_momentum

    # r = np.random.randint(0, star_count)
    # x[r] = np.random.uniform(0, 1.0)
    # y[r] = np.random.uniform(0, 1.0)
    # x_momentum[r] = 0
    # y_momentum[r] = 0

    # for z in range(0, 20):

    x_delta = np.zeros(star_count)
    y_delta = np.zeros(star_count)

    for i in range(0, star_count):
        xd = x[i] - x
        yd = y[i] - y
        xs = np.sign(xd)
        ys = np.sign(yd)
        d = (
                xd * xd +
                yd * yd
            ) + 0.000001
        attraction = (mass[i] + mass) / d * 0.00000001
        x_delta[i] = np.sum(xd * attraction)
        y_delta[i] = np.sum(yd * attraction)

    ambient_friction = 0.95
    x_momentum = x_delta + x_momentum * ambient_friction
    y_momentum = y_delta + y_momentum * ambient_friction

    x = x - x_momentum
    y = y - y_momentum

    # scale = 2
    # xs = (x - x_mass_center) * scale + width_2
    # ys = (y - y_mass_center) * scale + width_2
    # data = np.hstack((
    #     xs[:starCount_1, np.newaxis],
    #     ys[:starCount_1, np.newaxis]
    # ))

    x_mass_center = np.sum(x * mass_fraction)
    y_mass_center = np.sum(y * mass_fraction)

    x = (x - x_mass_center) + width_2
    y = (y - y_mass_center) + width_2

    # if frame % 10 == 0:

    data = np.hstack((
        x[:star_count_1, np.newaxis],
        y[:star_count_1, np.newaxis]
    ))

    scat.set_offsets(data)

    return scat,




def random_field():

    global x, y

    x = np.random.rand(star_count)
    y = np.random.rand(star_count)

def disc():

    global x, y

    polar_angle_xy = np.random.rand(star_count) * np.pi * 2

    mn = 0.0
    mx = 0.75
    md = mx - mn

    polar_distance = (np.random.rand(star_count) * md + mn) * width_2

    for p in range(0, star_count):
        x[p] = math.sin(polar_angle_xy[p]) * polar_distance[p] + width_2
        y[p] = math.cos(polar_angle_xy[p]) * polar_distance[p] + width_2

def two():

    o = 1 # 0.5
    d = 0.25

    for p in range(0, star_count // 2):
        x[p] = width_2 - x[p] * o + d
        # y[p] = width_2 - y[p] * o

    for p in range(star_count // 2, star_count):
        x[p] = width_2 + x[p] * o - d
        # y[p] = width_2 + y[p] * o

def hoedown():

    global x, y

    y = np.random.rand(star_count)

    o = 0.1

    for p in range(0, star_count // 2):
        x[p] = np.random.rand(1) * o

    for p in range(star_count // 2, star_count):
        x[p] = width - np.random.rand(1) * o


def uniform():

    global x, y

    x = np.random.uniform(0.0, 1.0, star_count)
    y = np.random.uniform(0.0, 1.0, star_count)