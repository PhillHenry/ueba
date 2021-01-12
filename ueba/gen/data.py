import random as r

import numpy as np


def square_data(N, period, noise=0):
    xf = np.random.rand(N, N)
    event = 1.
    offset = 1
    for i in range(N // period):
        row_idx = offset + (i * period)
        xf[row_idx, :N] = event
        for _ in range(noise):
            xf[row_idx, r.randint(0, N - 1)] = 0
    return xf

