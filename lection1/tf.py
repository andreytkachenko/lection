import mnist
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

np.set_printoptions(suppress=True)


def convert(y):
    y_d = np.zeros((len(y), 10))

    for idx, val in enumerate(y):
        y_d[idx, val] = 1.0

    return y_d


model = Sequential([
    Dense(32, activation='sigmoid', input_shape=(784, )),
    Dense(10, activation='sigmoid'),
])

model.compile(
    loss="mse",
    optimizer="sgd",
    metrics=['accuracy'])

(X, orig_y, val_X, orig_val_y) = mnist.load()

# X = (X / 255.0) / 2 + 0.5
# X = (X / 255.0) * 2 - 1

y = convert(orig_y)
val_y = convert(orig_val_y)

model.fit(X, y, epochs=15, batch_size=200)

print(model.evaluate(val_X, val_y))

# print(model.predict(np.array([X[0]]))[0])
# print(y[0])



