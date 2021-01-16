import numpy as np

import ueba.gen.vae_periodicity as to_test


n = 10
m = 10
ones = np.full(n, 1)
zeros = np.full(m, 0)


def test_matching_exactly_opposite():
    xs = np.hstack([ones, zeros])
    ys = np.hstack([zeros, ones])
    assert(to_test.matches(xs, ys) == n + m)


def test_matching_exactly_matching():
    ys = np.hstack([zeros, ones])
    xs = np.hstack([zeros, ones])
    assert(to_test.matches(xs, ys) == n + m)


def interleave(xs, ys):
    cs = np.empty(len(xs) + len(ys))
    cs[0::2] = xs  # 0::2 means "starting at 0, proceed in steps of 2"
    cs[1::2] = ys
    return cs


def test_matching_exactly_opposite_interleaved():
    xs = interleave(ones, zeros)
    ys = interleave(ones, zeros)
    assert(to_test.matches(xs, ys) == n + m)


def test_upto_half_wrong():
    for i in range(len(ones) // 2):
        assert_wrong(i + 1)


def test_both_zeros_is_valid():
    assert(to_test.matches(zeros, zeros) == len(zeros))


def test_over_half_wrong_switches_classes():
    xs = ones
    half = len(ones) // 2
    for i in range(half):
        num_wrong = i + half
        ys = xs.copy()
        ys[0:num_wrong] = 0
        assert(to_test.matches(xs, ys) == half + i)


def assert_wrong(num_wrong):
    xs = ones
    ys = xs.copy()
    ys[0:num_wrong] = 0
    assert(to_test.matches(xs, ys) == len(xs) - num_wrong)


