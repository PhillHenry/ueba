# LSTM example from https://machinelearningmastery.com/lstm-autoencoders/

import numpy as np
from math import pi
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
import matplotlib.pyplot as plt


def raw_data(n_cycles, n_points):
    return np.sin(np.arange(n_points) * (n_cycles * 2 * pi / n_points))


def reshape(xs):
    return xs.reshape((1, (len(xs)), 1))


def lstm_model(n_in):
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_in, 1)))
    model.add(RepeatVector(n_in))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(cycles, cycles, epochs=300, verbose=0)
    return model


# define input sequence
n_in = 50
n_samples = 10

cycles = raw_data(3, n_in)
for i in range (n_samples - 1):
    data = raw_data((i / n_samples) + 1, n_in)
    cycles = np.concatenate((cycles, data), axis=None)

# reshape input into [samples, timesteps, features]
cycles = cycles.reshape(n_samples, n_in, 1)

cycles_x1 = massage.rotate(raw_data(1, n_in), int(n_in / 5))
actual = reshape(np.copy(cycles_x1))

model = lstm_model(n_in)
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
yhat = model.predict(actual, verbose=0)
predictions = yhat[0, :, 0]
print(predictions)

plt.plot(cycles_x1)
plt.plot(predictions)
plt.show()
