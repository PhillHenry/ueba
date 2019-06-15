from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def data(n, t):
    d = np.random.rand(n, t)
    return d


if __name__ == "__main__":
    people = 100
    days = 365
    d = data(people, days)
    x = range(people)
    y = range(days)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X.T, Y.T, d)

    plt.show()
