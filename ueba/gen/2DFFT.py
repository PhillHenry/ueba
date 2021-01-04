from scipy.fft import ifftn, fftn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# from https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
def data(N):
    xf = np.random.rand(N, N)
    event = 1.
    period = 4
    offset = 1
    for i in range(N // period):
        xf[offset + (i * period), :N] = event
    return xf


N = 30
f, (ax1, ax2, ax3) = plt.subplots(3, 1)
xf = data(N)

xf -= xf.mean()
Z = np.fft.fftn(xf)

ax1.imshow(xf, cmap=cm.Reds)
ax2.imshow(np.real(Z), cmap=cm.gray)

dt = 0.1
fr = np.fft.fftfreq(N)
ax3.plot(np.fft.fftshift(fr), np.fft.fftshift(np.abs(Z)))
plt.show()
