import pylab as plt
import numpy as np
import seaborn as sns; sns.set()

import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.cm as cm

import data as d


def vector_shape(n):
    return (n * n,)


def sample(n, period):
    raw = d.square_data(n, period)
    v = np.reshape(raw, vector_shape(n))
    return v


def samples(num, d, period):
    xs = []
    for _ in range(num):
        xs.append(sample(d, period))
    return np.vstack(xs)


sample_size = 200
n = 28
period = 5
x_train = samples(sample_size, n, period)
x_test = samples(sample_size, n, period)

m = Sequential()
m.add(Dense(512,  activation='elu', input_shape=vector_shape(n)))
m.add(Dense(128,  activation='elu'))
m.add(Dense(2,    activation='linear', name="bottleneck"))
m.add(Dense(128,  activation='elu'))
m.add(Dense(512,  activation='elu'))
m.add(Dense(784,  activation='sigmoid'))
m.compile(loss='mean_squared_error', optimizer = Adam())

print("x_train shape = {},  x_test shape = {}".format(np.shape(x_train), np.shape(x_test)))

history = m.fit(x_train, x_train, batch_size=128, epochs=5, verbose=1,
                validation_data=(x_test, x_test))

encoder = Model(m.input, m.get_layer('bottleneck').output)
Zenc = encoder.predict(x_train)  # bottleneck representation
Renc = m.predict(x_train)        # reconstruction

print("Renc shape   = {}".format(np.shape(Renc)))
print("Renc         \n{}".format(Renc))

x = np.shape(Zenc)[0]
plt.subplot(121)
plt.title('bottleneck representation')
plt.scatter(Zenc[:x, 0], Zenc[:x, 1], s=8, cmap='tab10')

plt.subplot(122)
arbitrary = np.reshape(Renc[0], [n, n])
plt.title('random reconstruction')
plt.imshow(arbitrary, cmap=cm.Reds)

plt.tight_layout()
plt.show()

