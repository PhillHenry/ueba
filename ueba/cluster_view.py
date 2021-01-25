import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import string
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def outliers(coords, eps=0.4, min_samples=20):
    scanned = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    categorized = list(zip(scanned.labels_, coords))
    return list(filter(lambda x: x[0] == -1, categorized))


def find_outliers_in(data):
    num_features = np.shape(data)[1]
    index = "index"
    coord_columns = [x for x in string.ascii_lowercase[:num_features - 1]]
    columns = [index] + coord_columns
    df = pd.DataFrame(data[1:, 0:num_features], columns=columns)
    outlying = list(map(lambda x: x[1][0], outliers(data[1:, 1:num_features], 7000, 2)))
    outlier_lines = [int(x) for x in df[df[coord_columns[0]].isin(outlying)][index]]
    print("0-based lines of outliers = {}".format(outlier_lines))


def plot_3d(data):
    """
    :param data: one dimension of index, 3 of co-ordinates. Assumes 2 classes of equal size and sequential in array.
    """
    n = np.shape(data)[0] - 1
    fig = pyplot.figure()
    fig.set_tight_layout(False)
    ax = Axes3D(fig)
    cs = np.zeros(n, dtype=int)
    cs[:n//2] = 0
    cs[n//2:] = 1
    colors = np.array(["red", "green"])
    ax.scatter(data[1:, 1], data[1:, 2], data[1:, 3], c=colors[cs])
    pyplot.show()


if __name__ == "__main__":
    """
    Takes one argument that is the CSV file of the points co-ordinates. First column is the index. First row is the column names.
    """
    my_data = np.genfromtxt(sys.argv[1], delimiter=',')
    find_outliers_in(my_data)
    if np.shape(my_data)[1] == 4:
        plot_3d(my_data)
