import pylab as plt
import numpy as np
import seaborn as sns; sns.set()

import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam

import data as d


def vector_shape(n):
    return (n * n,)


def fake_data(n, period):
    return samples(100, n, period)


def sample(n, period):
    raw = d.square_data(n, period)
    v = np.reshape(raw, vector_shape(n))
    return v


def samples(num, d, period):
    xs = []
    for _ in range(num):
        xs.append(sample(d, period))
    return np.vstack(xs)


n = 28
period = 5
x_train = fake_data(n, period)
x_test = fake_data(n, period)
y_train = fake_data(n, period)

m = Sequential()
m.add(Dense(512,  activation='elu', input_shape=vector_shape(n)))
m.add(Dense(128,  activation='elu'))
m.add(Dense(2,    activation='linear', name="bottleneck"))
m.add(Dense(128,  activation='elu'))
m.add(Dense(512,  activation='elu'))
m.add(Dense(784,  activation='sigmoid'))
m.compile(loss='mean_squared_error', optimizer = Adam())

print("x_train shape = {},  x_test shape = {}".format(np.shape(x_train), np.shape(x_test)))

history = m.fit(x_train, x_train, batch_size=128, epochs=5000, verbose=1,
                validation_data=(x_test, x_test))

encoder = Model(m.input, m.get_layer('bottleneck').output)
Zenc = encoder.predict(x_train)  # bottleneck representation
Renc = m.predict(x_train)        # reconstruction

plt.title('Autoencoder')
x = 100
plt.scatter(Zenc[:x,0], Zenc[:x,1], s=8, cmap='tab10')
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])

plt.tight_layout()
plt.show()

