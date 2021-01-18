import sys
import ueba.gen.vae_periodicity as vae
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import pylab as plt
from random import shuffle
from functools import reduce


def ngrams_of(words, n=2):
    xss = [words[i:] for i in range(n)]
    return reduce(lambda acc, x: ["".join((a, b)) for a, b in zip(acc, x) ], xss)


def all_ngrams_of(words, n):
    ngrams = []
    for i in range(2, n+1):
        ngrams += ngrams_of(words, i)
    return ngrams


def enhance(lines, n):
    enhanced = []
    for line in lines:
        words = line.split()
        words = words + all_ngrams_of(words, n)
        enhanced.append(" ".join(words))
    return enhanced


def index_words(lines, word_index={}):
    max_index = 0
    for line in lines:
        words = line.split()
        for x in words:
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
    history = m.fit(x_train, x_train, batch_size=2, epochs=200, verbose=1, validation_data=(x_test, x_test))
    encoder = Model(m.input, m.get_layer('bottleneck').output)
    return history, encoder


def max_length_of(xs):
    max_length = 0
    for x in xs:
        max_length = max(max_length, len(x))
    return max_length


def plot_loss(history, truncation=0):
    plt.plot(history.history['loss'][truncation:])
    plt.plot(history.history['val_loss'][truncation:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs (from epoch {})'.format(truncation))
    plt.legend(['train', 'test'], loc='upper right')


def run(lines1, lines2, max_vector_length):

    dictionary = index_words(lines1)
    dictionary = index_words(lines2, dictionary)

    vector1 = vectorize(lines1, dictionary)
    vector2 = vectorize(lines2, dictionary)

    vec_length = min(max_vector_length, max(max_length_of(vector1), max_length_of(vector2)))
    print("vec_length = {}".format(vec_length))
    vectors1 = truncate_or_pad(vector1, vec_length)
    vectors2 = truncate_or_pad(vector2, vec_length)
    shuffle(vectors1)
    shuffle(vectors2)

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
    plot_loss(history, 20)
    plt.savefig("/tmp/bash_history.png")
    plt.show()


if __name__ == "__main__":
    xs = read(sys.argv[1])
    ys = read(sys.argv[2])
    xs_enhanced = enhance(xs, 3)
    ys_enhanced = enhance(ys, 3)
    run(xs_enhanced, ys_enhanced, 256)

