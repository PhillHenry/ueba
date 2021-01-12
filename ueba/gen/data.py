import random as r

import numpy as np


event = 1.


def square_data(N, period, noise=0):
    xf = np.random.rand(N, N)
    offset = 1
    for i in range(N // period):
        row_idx = offset + (i * period)
        xf[row_idx, :N] = event
        for _ in range(noise):
            xf[row_idx, r.randint(0, N - 1)] = 0
    return xf


def square_data_mixed_periods(N, max_period, noise=0):
    periods = list(map(lambda x: int(x), np.random.uniform(1, max_period, N)))
    xs = np.zeros([N, N])
    for i in range(N):
        for (j, period) in enumerate(periods):
            modded = (i + 1) % period
            if modded == 0:
                xs[i, j] = event
    return xs


if __name__ == "__main__":
    print(square_data_mixed_periods(10, 3))