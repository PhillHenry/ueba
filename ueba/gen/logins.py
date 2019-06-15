from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


def weekends(d):
    for i in range(d.shape[1]):
        if i%7 == 0 or i%7 == 1:
            d[:, i] = 0


def data(n, t):
    d = np.random.rand(n, t)
    weekends(d)
    return d


if __name__ == "__main__":
    people = 100
    days = 21
    d = data(people, days)
    x = range(people)
    y = range(days)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X.T, Y.T, d)

    plt.show()
