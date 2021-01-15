import random as r

import numpy as np
import pylab as plt
import matplotlib.cm as cm

event = 1.


def create_random_square(n):
    return np.random.rand(n, n)


def square_data(n, period, noise=0):
    xf = create_random_square(n)
    offset = 1
    top = max(n, n // period + 1)
    for i in range(top):
        row_idx = offset + (i * period)
        if row_idx < n:
            xf[row_idx, :n] = event
            for _ in range(noise):
                xf[row_idx, r.randint(0, n - 1)] = 0
    return xf


def create_noise(xs, noise):
    N = len(xs)
    for _ in range(noise):
        random_x = r.randint(0, N - 1)
        xs[random_x] = 0


def square_data_mixed_periods(n, max_period, noise):
    periods = list(map(lambda x: int(x), np.random.uniform(1, max_period, n)))
    xs = create_random_square(n)
    for i in range(n):
        for (j, period) in enumerate(periods):
            modded = (i + 1) % period
            if modded == 0:
                xs[i, j] = event
        create_noise(xs[i,:], noise)
    return xs


if __name__ == "__main__":
    noise = 5
    period = 5
    n = 28
    for i in range(3):
        xs = square_data(n, period, noise)
        plt.subplot(231 + i)
        plt.imshow(xs, cmap=cm.Reds)
    for i in range(3):
        xs = square_data_mixed_periods(n, period, noise)
        plt.subplot(234 + i)
        plt.imshow(xs, cmap=cm.Reds)
    plt.figtext(.6, .47, "Regular activity with noise")
    plt.figtext(.6, .95, "Mixed periods with noise")
    plt.savefig("/tmp/data.png")
    plt.show()
