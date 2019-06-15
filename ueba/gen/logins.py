from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def data(n, t):
    d = np.random.rand(n, t)
    return d


if __name__ == "__main__":
    people = 100
    days = 100
    d = data(people, days)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = range(days)
    ys = range(people)
    X, Y = np.meshgrid(xs, ys)  # `plot_surface` expects `x` and `y` data to be 2D
    ax.plot_surface(X, Y, data)

    plt.show()
