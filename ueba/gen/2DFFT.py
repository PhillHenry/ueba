from scipy.fft import ifftn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# from https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
N = 30
f, (ax1, ax4) = plt.subplots(2, 1, sharex='col', sharey='row')
xf = np.zeros((N,N))
event = 1.
xf[0, :N] = event
xf[7, :N] = event
xf[14, :N] = event
xf[21, :N] = event
xf[28, :N] = event
Z = ifftn(xf)
ax1.imshow(xf, cmap=cm.Reds)
ax4.imshow(np.real(Z), cmap=cm.gray)
plt.show()
