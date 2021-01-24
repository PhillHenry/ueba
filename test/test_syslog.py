import ueba.parse.syslog as to_test


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
