from scipy.fft import ifftn, fftn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random as r


# from https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
def data(N, noise=0):
    xf = np.random.rand(N, N)
    event = 1.
    period = 8
    offset = 1
    for i in range(N // period):
        row_idx = offset + (i * period)
        xf[row_idx, :N] = event
        for _ in range(noise):
            xf[row_idx, r.randint(0, N - 1)] = 0
    return xf


def plot(raw, frequencies):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fr = np.fft.fftfreq(frequencies.shape[0])
    xticks = np.fft.fftshift(fr)

    ax1.imshow(raw, cmap=cm.Reds)
    ax2.imshow(frequencies)
    tick_range = np.arange(min(xticks), max(xticks), 0.2)
    tick_labels = list(map(lambda x: '%.2f' % x, tick_range))
    print(tick_labels)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticklabels(tick_labels)

    ax3.plot(xticks, frequencies)
    plt.show()


def frequencies_for(N):
    raw = data(N, 15)
    raw -= raw.mean()
    Z = np.fft.fftn(raw)
    frequencies = np.fft.fftshift(np.abs(Z))
    plot(raw, frequencies)
    return frequencies


def find_signal_in_fake_data():
    frequencies = frequencies_for(40)
    max_t = float("-inf")
    coords = None
    x_range = range(1, frequencies.shape[0] - 1)
    for i in x_range:
        for j in x_range:
            if frequencies[i, j] > max_t:
                coords = [i, j]
                max_t = frequencies[i, j]
    fr = np.fft.fftfreq(frequencies.shape[0])
    freq = fr[coords[0]]
    print("Max value of {} at {} corresponding to frequency {}".format(max_t, coords, freq))


if __name__ == "__main__":
    find_signal_in_fake_data()

