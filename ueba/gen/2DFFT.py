import random as r

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import data as d


# from https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
def ticks_for(frequencies):
    fr = np.fft.fftfreq(frequencies.shape[0])
    return np.fft.fftshift(fr)


def plot(raw, frequencies):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1)
    xticks = ticks_for(frequencies)

    ax1.imshow(raw, cmap=cm.Reds)
    ax2.imshow(frequencies)
    tick_range = np.arange(min(xticks), max(xticks), 0.2)
    tick_labels = list(map(lambda x: '%.2f' % x, tick_range))
    print(tick_labels)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticklabels(tick_labels)

    ax3.plot(xticks, frequencies)
    plt.show()


def frequencies_for(N, period):
    raw = d.square_data(N, period, 24)
    raw -= raw.mean()
    Z = np.fft.fftn(raw)
    frequencies = np.fft.fftshift(np.abs(Z))
    plot(raw, frequencies)
    return frequencies


def find_signal_in_fake_data():
    N = 64
    period = 8
    frequencies = frequencies_for(N, period)
    max_t = float("-inf")
    coords = None
    x_range = range(1, frequencies.shape[0] - 1)
    for i in x_range:
        for j in x_range:
            f = frequencies[i, j]
            if f > max_t:
                coords = [i, j]
                max_t = f
    analyse(coords, frequencies, max_t, N, period)


def analyse(coords, frequencies, max_t, N, period):
    indx_row_w_max = coords[0]
    ticks = ticks_for(frequencies)
    freq_of_max = ticks[indx_row_w_max]
    print("Max value of {} at {} corresponding to frequency {}".format(max_t, coords, freq_of_max))
    column = frequencies[:, coords[1]]
    f_to_coords = zip(column, [(i, coords[1]) for i in range(N)])
    peaks = list(filter(lambda x: x[0] > (0.8 * max_t), f_to_coords))
    print("peaks = {}".format(peaks))
    points_max = list(map(lambda x: x[1][0], peaks))
    freqs_w_max_amplitude = set(map(lambda x: abs(x), ticks[points_max]))
    print("potential frequencies: {}".format(freqs_w_max_amplitude))
    candidate = min(filter(lambda x: x > 0, freqs_w_max_amplitude))
    print("discovered value = {} actual value = {}".format(1./candidate, period))


if __name__ == "__main__":
    find_signal_in_fake_data()

