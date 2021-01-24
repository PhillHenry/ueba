import ueba.parse.syslog as to_test
import numpy as np


def create_logs(n=20, offset=42, label_cardinality=7):
    return [to_test.SyslogEntry(i + offset, "label{}".format(i % label_cardinality)) for i in range(n)]


def test_parse_syslog_time():
    assert(to_test.parse_to_epoch('Jan 22 14:41:41') == -2207121499.0)


def test_normalize_time():
    xs = create_logs()
    to_test.normalize_time(xs)
    assert(to_test.min_time_of(xs) == 0)


def test_vectorize():
    n = 29
    label_cardinality = 7
    xs = create_logs(label_cardinality=label_cardinality, n=n)
    points, _ = to_test.vectorize(xs)
    assert(len(points) == len(xs))
    labels = set(map(lambda x: x[0], points))
    assert(len(labels) == label_cardinality)
    times = list(map(lambda x: x[1], points))
    assert(min(times) == 0)
    assert(max(times) == n-1)


def test_labels_to_maxes():
    maxes = [101, 102, 103, 104, 105]
    label_to_index = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    label_to_max = to_test.labels_to_maxes(maxes, label_to_index)
    assert(label_to_max["a"] == 101)
    assert(label_to_max["b"] == 102)
    assert(label_to_max["c"] == 103)
    assert(label_to_max["d"] == 104)
    assert(label_to_max["e"] == 105)


def test_max_in_columns_of():
    m = np.reshape(np.arange(100), [10, 10])
    maxes = to_test.max_in_columns_of(m)
    expected = list(range(90, 100))
    assert(np.array_equal(maxes, expected))
