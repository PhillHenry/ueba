import sys
import ueba.gen.vae_periodicity as vae
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import pylab as plt


def index_words(lines, word_index={}):
    max_index = 0
    for line in lines:
        xs = line.split()
        for x in xs:
            if word_index.get(x) is None:
                word_index[x] = max_index
                max_index += 1
    return word_index


def vectorize(lines, word_index):
    vectors = []
    for line in lines:
        xs = line.split()
        vectors.append([word_index[x] for x in xs])
    return vectors


def truncate_or_pad(vs, n):
    sized = []
    for v in vs:
        length = len(v)
        if length > n:
            sized.append(v[:n])
        else:
            padding = [0] * (n - length)
            v = v + padding
            sized.append(v)
    return sized


def read(file):
    lines = []
    for line in open(file, "r"):
        lines.append(line)
    return lines


def create_model(n):
    m = Sequential()
    hidden_size = 3 * n//4
    m.add(Dense(n,              activation='elu', input_shape=(n,)))
    m.add(Dense(hidden_size,    activation='elu'))
    m.add(Dense(2,              activation='linear', name="bottleneck"))
    m.add(Dense(hidden_size,    activation='elu'))
    m.add(Dense(n,              activation='elu'))
    # m.add(Dense(n,              activation='sigmoid'))
    m.compile(loss='mean_squared_error', optimizer=Adam())
    return m


def train(vectors):
    vec_length = len(vectors[0])
    vectors = np.vstack(vectors)
    m = create_model(vec_length)
    train_size = int(len(vectors) * 0.5)
    x_train = vectors[:train_size]
    x_test = vectors[train_size:]
    print("Train size = {}, test size = {}, vector length = {}".format(len(x_train), len(x_test), vec_length))
    history = m.fit(x_train, x_train, batch_size=4, epochs=100, verbose=1, validation_data=(x_test, x_test))
    encoder = Model(m.input, m.get_layer('bottleneck').output)
    return history, encoder


def run(lines1, lines2):
    vec_length = 16

    dictionary = index_words(lines1)
    dictionary = index_words(lines2, dictionary)
    vectors1 = truncate_or_pad(vectorize(lines1, dictionary), vec_length)
    vectors2 = truncate_or_pad(vectorize(lines2, dictionary), vec_length)

    history, encoder = train(vectors1)
    vec1_representation = encoder.predict(np.vstack(vectors1))
    vec2_representation = encoder.predict(np.vstack(vectors2))
    mixed = np.vstack([vec1_representation, vec2_representation])
    n_total = len(mixed)
    ys = np.zeros([n_total, ])
    ys[len(vectors1):] = 1

    plt.subplot(211)
    vae.plot_clusters(mixed, n_total, ys)

    plt.subplot(212)
    vae.plot_loss(history)
    plt.show()


if __name__ == "__main__":
    xs = read(sys.argv[1])
    ys = read(sys.argv[2])
    run(xs, ys)

