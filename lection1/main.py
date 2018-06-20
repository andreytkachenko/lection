import numpy as np
import mnist

np.random.seed(1)


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1.0 / (1.0 + np.exp(-x))


# Считаем точность
def calc_accuracy(y, res):
    wrong = 0.0
    total = len(y)
    for idx, r in enumerate(res):
        if np.argmax(r) != y[idx]:
            wrong += 1.0

    return 1.0 - (wrong / total)


# `orig_y` выглядит так [2, 8, 1, 5 ....]
# а нам нужно:
# [[0, [0, [0, [0, ....]
#   0,  0,  1,  0,
#   1,  0,  0,  0,
#   0,  0,  0,  0,
#   0,  0,  0,  0,
#   0,  0,  0,  1,
#   0,  0,  0,  0,
#   0,  0,  0,  0,
#   0,  1,  0,  0,
#   0], 0], 0], 0],

def convert(y):
    y_d = np.zeros((len(y), 10))

    for idx, val in enumerate(y):
        y_d[idx, val] = 1.0

    return y_d


(X, orig_y, val_X, orig_val_y) = mnist.load()

y = convert(orig_y)
val_y = convert(orig_val_y)

hidden_count = 32

W1 = np.random.normal(0.0, 0.1, (784, hidden_count))
b1 = np.zeros((hidden_count, ))

W2 = np.random.normal(0.0, 0.1, (hidden_count, 10))
b2 = np.zeros((10, ))

lr = 0.01
alpha = 0.1

print('X:  ', X.shape)
print('y:  ', y.shape)

print('W1: ', W1.shape)
print('b1: ', b1.shape)

print('W2: ', W2.shape)
print('b2: ', b2.shape)

z = sigmoid(X.dot(W1) + b1)
_y = sigmoid(z.dot(W2) + b2)

print('Точность: ', calc_accuracy(orig_y, _y) * 100, '%')


batch_size = 70
batch_count = 60_000 // batch_size

pdW1 = W1.copy()
pdW2 = W2.copy()
pdb1 = b1.copy()
pdb2 = b2.copy()

pdW1.fill(0)
pdW2.fill(0)
pdb1.fill(0)
pdb2.fill(0)

for i in range(0, 10):
    for batch_idx in range(0, batch_count):
        begin = batch_idx * batch_size
        end = begin + batch_size

        batch_X = X[begin:end]
        batch_y = y[begin:end]

        factor = 1.0 / len(batch_X)

        # forward

        l1 = sigmoid(batch_X.dot(W1) + b1)
        l2 = sigmoid(l1.dot(W2) + b2)

        # backward

        l2_error = l2 - batch_y
        l2_delta = l2_error * sigmoid(l2, True)

        l1_error = l2_delta.dot(W2.T)
        l1_delta = l1_error * sigmoid(l1, True)

        dW2 = l1.T.dot(l2_delta) * factor
        db2 = np.mean(l2_delta, axis=0)

        dW1 = batch_X.T.dot(l1_delta) * factor
        db1 = np.mean(l1_delta, axis=0)

        wd = 0.0
        alpha = 0.1
        mu = 0.1

        W1 -= lr * dW1
        W2 -= lr * dW2
        b1 -= lr * db1
        b2 -= lr * db2

        #
        # prevW1 = pdW1.copy()
        # prevW2 = pdW2.copy()
        # prevb1 = pdb1.copy()
        # prevb2 = pdb2.copy()
        #
        # pdW1[:] = dW1 * mu + pdW1 * alpha
        # pdW2[:] = dW2 * mu + pdW2 * alpha
        # pdb1[:] = db1 * mu + pdb1 * alpha
        # pdb2[:] = db2 * mu + pdb2 * alpha
        #
        # W1 -= (wd * np.mean(W1)) + prevW1 * (-alpha) + pdW1 * (1.0 + alpha)
        # W2 -= (wd * np.mean(W2)) + prevW2 * (-alpha) + pdW2 * (1.0 + alpha)
        # b1 -= (wd * np.mean(b1)) + prevb1 * (-alpha) + pdb1 * (1.0 + alpha)
        # b2 -= (wd * np.mean(b2)) + prevb2 * (-alpha) + pdb2 * (1.0 + alpha)

    z = sigmoid(val_X.dot(W1) + b1)
    _y = sigmoid(z.dot(W2) + b2)

    print(i, 'Точность: ', calc_accuracy(orig_val_y, _y) * 100, '%')
