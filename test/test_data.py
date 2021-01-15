import numpy as np

import ueba.gen.data as to_test

n = 100
noise = 20


def test_square_data_mixed_periods():
    max_period = 10
    m = to_test.square_data_mixed_periods(n, max_period, noise)
    assert(m.shape == (n, n))
    print(m)
    assert(number_of_zeros_in(np.reshape(m, [n*n])) > noise)


def test_create_noise():
    xs = np.full(n, 1)
    to_test.create_noise(xs, noise)
    assert(number_of_zeros_in(xs) > noise / 2)  # some random setting to 0 will be duped


def number_of_zeros_in(xs):
    return len(list(filter(lambda x: x == 0, xs)))
