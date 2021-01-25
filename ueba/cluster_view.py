import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import string


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


if __name__ == "__main__":
    my_data = np.genfromtxt(sys.argv[1], delimiter=',')
    find_outliers_in(my_data)
