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
        items = line.split(" ")
        date = " ".join(items[:3])
        epoch = parse_to_epoch(date)
        strip_pid = re.compile(r"\[.*", re.IGNORECASE)
        label = strip_pid.sub("", items[4])
        entry = SyslogEntry(epoch, label)
        parsed.append(entry)
    return parsed


def parse_to_epoch(date):
    return time.mktime(datetime.datetime.strptime(date, "%b %d %I:%M:%S").timetuple())


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
            print(x.label)
            label_to_index[x.label] = label_counter
            label_counter += 1
        points.append([label_to_index[x.label], x.time])
    return points


def as_matrix(filename, max_y=200):
    parsed = parse(filename)
    points = vectorize(parsed)
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


def fourier_of(filename):
    raw = as_matrix(filename).transpose()
    raw -= raw.mean()
    Z = np.fft.fftn(raw)
    frequencies = np.fft.fftshift(np.abs(Z))
    f, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(raw, cmap=cm.Reds)

    xticks = ticks_for(frequencies)
    ax2.plot(xticks, frequencies)

    plt.show()


if __name__ == "__main__":
    fourier_of(sys.argv[1])
