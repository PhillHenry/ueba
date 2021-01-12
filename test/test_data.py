import numpy as np
import math as math
import pytest
import ueba.gen.data as to_test


def test_square_data_mixed_periods():
    N = 100
    max_period = 10
    m = to_test.square_data_mixed_periods(N, max_period)
    assert(m.shape == (N, N))
    print(m)
