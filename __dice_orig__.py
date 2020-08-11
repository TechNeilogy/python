import numpy as np
import matplotlib.pyplot as plt


faces = 6
trials = 1000000
min_die = 70
max_die = 140
skip = 4


faces1 = faces + 1

fig, ax = plt.subplots()
ax.set_title("{0:,} Rolls of {1}-{2} Die with {3} Faces".format(trials, min_die, max_die, faces))
ax.set_xlabel('Sum of Faces')
ax.set_ylabel('Observations')


for dice in range(min_die, max_die+1, skip):

    possible1 = faces * dice + 1

    sums = np.zeros(possible1)

    for trial in range(0, trials):
        sums[np.random.randint(1, faces1, dice).sum()] += 1

    plt.plot(np.arange(dice, possible1), sums[dice:], label=dice)

legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
legend.set_title("# of Die")

plt.show()