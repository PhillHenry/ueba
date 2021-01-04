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
        for _ in range(N):
            xf[row_idx, r.randint(0, N - 1)] = 0
    return xf


N = 40
f, (ax1, ax2, ax3) = plt.subplots(3, 1)
xf = data(N, 3)

xf -= xf.mean()
Z = np.fft.fftn(xf)
frequencies = np.fft.fftshift(np.abs(Z))
fr = np.fft.fftfreq(N)
xticks = np.fft.fftshift(fr)

ax1.imshow(xf, cmap=cm.Reds)
ax2.imshow(frequencies)
tick_range = np.arange(min(xticks), max(xticks), 0.2)
tick_labels = list(map(lambda x: '%.2f' % x, tick_range))
print(tick_labels)
ax2.set_xticklabels(tick_labels)
ax2.set_yticklabels(tick_labels)

ax3.plot(xticks, frequencies)
plt.show()
