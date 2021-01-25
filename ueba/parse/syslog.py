import time
import datetime
import sys
import ueba.parse.files as fi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re


class SyslogEntry:
    def __init__(self, timestr, label):
        self.time = int(timestr)
        self.label = label


def parse(filename):
    lines = fi.read(filename)
    parsed = []
    for line in lines:
        items = list(filter(lambda x: x != "", line.split(" ")))
        date = " ".join(items[:3])
        epoch = parse_to_epoch(date)
        strip_pid = re.compile(r"\[.*", re.IGNORECASE)
        label = strip_pid.sub("", items[4])
        if label.startswith("kernel") and "UFW" in items[6]:
            label += items[6]
        entry = SyslogEntry(epoch, label)
        parsed.append(entry)
    return parsed


def parse_to_epoch(date):
    return time.mktime(datetime.datetime.strptime(date, "%b %d %H:%M:%S").timetuple())


def min_time_of(logs):
    return min(map(lambda x: x.time, logs))


def normalize_time(logs):
    min_time = min_time_of(logs)
    for x in logs:
        x.time = x.time - min_time
    return logs


def vectorize(logs):
    normalize_time(logs)
    label_counter = 0
    label_to_index = {}
    points = []
    for x in logs:
        if label_to_index.get(x.label) is None:
            label_to_index[x.label] = label_counter
            label_counter += 1
        points.append([label_to_index[x.label], x.time])
    return points, label_to_index


def as_matrix(points, max_y=200):

    max_time = max(map(lambda x: x[1], points))
    labels = set(map(lambda x: x[0], points))
    max_x = len(labels)

    print("number of points = {}, max_time = {}, labels = {}".format(len(points), max_time, labels))

    raw = np.zeros([max_x, max_y + 1])

    for pt in points:
        y = int(pt[1] * max_y / max_time)
        x = pt[0]
        raw[x, y] = raw[x, y] + 1

    return raw


# from 2DFFT - TODO change that filename so we can import it
def ticks_for(frequencies):
    fr = np.fft.fftfreq(frequencies.shape[0])
    return np.fft.fftshift(fr)


def fourier_of(raw):
    raw -= raw.copy().mean()
    Z = np.fft.fftn(raw)
    frequencies = np.fft.fftshift(np.abs(Z))
    return frequencies


def max_in_columns_of(m):
    maxes = []
    for col_idx in range(np.shape(m)[1]):
        maxes.append(max(m[:,col_idx]))
    return maxes


def index_to_labels(label_to_index):
    return dict((v, k) for k, v in label_to_index.items())


def infrequent(threshold, maxes, label_to_index):
    max_freq = max(maxes)
    index_to_label = index_to_labels(label_to_index)
    irregular_labels = []
    for i, mx in enumerate(maxes):
        if mx < threshold * max_freq:
            label = index_to_label[i]
            irregular_labels.append(label)
    return irregular_labels


def labels_to_maxes(maxes, label_to_index):
    index_to_label = index_to_labels(label_to_index)
    label_to_max = {}
    for i, mx in enumerate(maxes):
        label = index_to_label[i]
        label_to_max[label] = mx
    return label_to_max


def sort_on_value(dictionary):
    kvs = [(k, v) for k, v in dictionary.items()]
    kvs.sort(key=lambda x: x[1])
    return kvs


def find_patterns_in(filename):
    parsed = parse(filename)
    points, label_to_index = vectorize(parsed)
    raw = as_matrix(points).transpose()
    frequencies = fourier_of(raw)

    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
    # plt.subplot(211)
    # ax1.imshow(raw, cmap=cm.Reds)
    plot_points = np.vstack(points)
    max_y = max(plot_points[:, 0])
    ax1.set_ylim(0, max_y)
    kvs = sort_on_value(label_to_index)
    y_labels = list(map(lambda x: x[0], kvs))
    # ax1.set_yticklabels(y_labels)
    ax1.scatter(plot_points[:, 1], plot_points[:, 0], s=2)
    ax1.set_title("Events")

    xticks = ticks_for(frequencies)
    ax2.plot(xticks, frequencies)
    ax2.set_title("Frequencies")

    maxes = max_in_columns_of(frequencies)

    label_to_max = labels_to_maxes(maxes, label_to_index)
    label_maxes = sort_on_value(label_to_max)
    print("syslog contributors (least regular to most regular):")
    for label, mx in label_maxes:
        print("%40s%20f" % (label, mx))

    plt.savefig("/tmp/syslog.png")
    plt.show()


if __name__ == "__main__":
    find_patterns_in(sys.argv[1])
