import pylab as plt
import numpy as np
import seaborn as sns; sns.set()

import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.cm as cm
from sklearn.cluster import KMeans

import data as d


# Inspired by https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca


def vector_shape(n):
    return (n * n,)


def square_to_vector(raw, n):
    return np.reshape(raw, vector_shape(n))


def sample(n, period):
    raw = d.square_data_mixed_periods(n, period)
    v = square_to_vector(raw, n)
    return v


def samples(num, d, period):
    xs = []
    for _ in range(num):
        xs.append(sample(d, period))
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


def plot(Zenc, Renc, x, ys, n):
    plt.subplot(311)
    plt.title('bottleneck representation')
    plt.scatter(Zenc[:x, 0], Zenc[:x, 1], c=ys, s=8, cmap='jet')

    plt.subplot(312)
    arbitrary = np.reshape(Renc[0], [n, n])
    plt.title('random reconstruction')
    plt.imshow(arbitrary, cmap=cm.Reds)

    return plt


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


def calc_accuracy(mixed, periodicals, baseline, ys):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(mixed)
    n_periodicals = np.shape(periodicals)[0]
    n_baseline = np.shape(baseline)[0]
    predicted_to_actual = list(zip(kmeans.labels_, ys))
    n_matching = sum(map(lambda x: x[0] * x[1], predicted_to_actual))
    labels_match = n_matching >= (n_baseline + n_periodicals) / 2
    results = list(map(lambda x: correct(labels_match, x), predicted_to_actual))
    n_correct = sum(results)

    print("accuracy {}".format(float(n_correct) / float(n_periodicals + n_baseline)))


def run():
    sample_size = 256
    n = 28
    period = 5
    x_train = samples(sample_size, n, period)
    x_test = samples(sample_size, n, period)

    m = create_model(vector_shape(n))

    print("x_train shape = {},  x_test shape = {}".format(np.shape(x_train), np.shape(x_test)))

    history = m.fit(x_train, x_train, batch_size=128, epochs=10, verbose=1,
                    validation_data=(x_test, x_test))

    encoder = Model(m.input, m.get_layer('bottleneck').output)
    periodicals = encoder.predict(x_train)  # bottleneck representation
    Renc = m.predict(x_train)        # reconstruction

    baseline = encoder.predict(randoms(sample_size, n))  # no periodicity
    mixed = np.vstack([periodicals, baseline])
    n_total = np.shape(mixed)[0]
    ys = np.zeros([n_total, ])
    ys[sample_size:] = 1

    calc_accuracy(mixed, periodicals, baseline, ys)

    plt = plot(mixed, Renc, n_total, ys, n)

    print(history.history.keys())

    # see https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    # summarize history for loss
    plt.subplot(313)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    run()
