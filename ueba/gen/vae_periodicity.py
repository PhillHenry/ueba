import pylab as plt
import numpy as np
import seaborn as sns; sns.set()

import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import pandas as pd

import ueba.gen.data as d


# Inspired by https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca


def vector_shape(n):
    return (n * n,)


def square_to_vector(raw, n):
    return np.reshape(raw, vector_shape(n))


def sample_mixed_periods(n, max_period, noise):
    raw = d.square_data_mixed_periods(n, max_period, noise)
    v = square_to_vector(raw, n)
    return v


def sample_fixed_period(n, max_period, noise):
    raw = d.square_data(n, max_period, noise)
    v = square_to_vector(raw, n)
    return v


def samples_fixed_period(num, n, period, noise):
    xs = []
    for _ in range(num):
        xs.append(sample_fixed_period(n, period, noise))
    return np.vstack(xs)


def samples(num, n, period, noise):
    xs = []
    for _ in range(num):
        xs.append(sample_mixed_periods(n, period, noise))
    return np.vstack(xs)


def randoms(num, n):
    xs = []
    for _ in range(num):
        xs.append(np.random.rand(n * n))
    return np.vstack(xs)


def create_model(shape):
    m = Sequential()
    m.add(Dense(512,  activation='elu', input_shape=shape))
    m.add(Dense(128,  activation='elu'))
    m.add(Dense(2,    activation='linear', name="bottleneck"))
    m.add(Dense(128,  activation='elu'))
    m.add(Dense(512,  activation='elu'))
    m.add(Dense(784,  activation='sigmoid'))
    m.compile(loss='mean_squared_error', optimizer=Adam())
    return m


def plot_clusters(Zenc, x, ys):
    plt.title('bottleneck representation')
    plt.scatter(Zenc[:x, 0], Zenc[:x, 1], c=ys, s=8, cmap='jet')


def plot_reconstruction(Renc, n):
    arbitrary = np.reshape(Renc[0], [n, n])
    plt.title('random reconstruction')
    plt.imshow(arbitrary, cmap=cm.Reds)


def correct(matching, x):
    if matching:
        if x[0] == x[1]:
            return 1
        else:
            return 0
    else:
        if x[0] != x[1]:
            return 1
        else:
            return 0


def calc_accuracy(mixed, ys):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(mixed)
    xs = kmeans.labels_

    n_correct = matches(xs, ys)

    print("accuracy {}".format(float(n_correct) / len(ys)))


def matches(xs, ys):
    assert(len(xs) == len(ys))
    n_total = len(xs)
    predicted_to_actual = list(zip(xs, ys))
    n_matching = sum(map(lambda x: 1 if x[0] == x[1] else 0, predicted_to_actual))
    labels_match = n_matching >= n_total / 2
    results = list(map(lambda x: correct(labels_match, x), predicted_to_actual))
    n_correct = sum(results)
    return n_correct


def plot_loss(history):
    # see https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')


def run():
    sample_size = 256
    n = 28
    period = 5
    noise = 3
    x_train = samples(sample_size, n, period, noise)
    x_test = samples(sample_size, n, period, noise)

    m = create_model(vector_shape(n))

    print("x_train shape = {},  x_test shape = {}".format(np.shape(x_train), np.shape(x_test)))

    history = m.fit(x_train, x_train, batch_size=128, epochs=10, verbose=1,
                    validation_data=(x_test, x_test))

    encoder = Model(m.input, m.get_layer('bottleneck').output)
    periodicals = encoder.predict(x_train)  # bottleneck representation
    Renc = m.predict(x_train)               # reconstruction

    baseline = encoder.predict(samples_fixed_period(sample_size, n, period, noise))  # no periodicity
    mixed = np.vstack([periodicals, baseline])
    n_total = np.shape(mixed)[0]
    ys = np.zeros([n_total, ])
    ys[sample_size:] = 1

    calc_accuracy(mixed, ys)

    plt.subplot(311)
    plot_clusters(mixed, n_total, ys)
    plt.subplot(312)
    plot_reconstruction(Renc, n)
    print(history.history.keys())
    pd.DataFrame(mixed).to_csv("/tmp/bottlenecked.csv")

    plt.subplot(313)
    plot_loss(history)

    plt.show()


if __name__ == "__main__":
    run()
