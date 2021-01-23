import time
import datetime
import sys
import ueba.parse.files as fi


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
        label = items[3]
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
            label_to_index[x.label] = label_counter
            label_counter += 1
        points.append([label_to_index[x.label], x.time])
    return points


if __name__ == "__main__":
    parse(sys.argv[1])
