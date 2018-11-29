"""produce an animation of the population over time"""

import numpy as np
import NSGA_II
import matplotlib.pyplot as plt
from matplotlib import animation

objectives = [lambda x: abs(x[0]), lambda y: 1 / (abs(y[0]) + 0.1) + y[1] ** 2]
bounds = [(-2, 2), (-2, 2)]
moo = NSGA_II.NSGA_II(objectives, bounds, iterations=10, parent_pop_size=100)
pareto_set = moo.run()
pop_storage = moo.population_storage
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    x =
