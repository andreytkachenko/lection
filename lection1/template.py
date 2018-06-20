import numpy as np
import mnist

# чтобы случайные числа генерировались одинаковые для каждого запуска программы
np.random.seed(1)


# Сигмоида
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Производная сигмоиды
def sigmoid_prime(z):
    return z * (1 - z)


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

W1 = (np.random.rand(784, 32) - 0.5) * 0.5
b1 = np.zeros((32,))

W2 = (np.random.rand(32, 10) - 0.5) * 0.5
b2 = np.zeros((10,))

print('X:  ', X.shape)
print('y:  ', y.shape)

print('W1: ', W1.shape)
print('b1: ', b1.shape)

print('W2: ', W2.shape)
print('b2: ', b2.shape)

# Backward
alpha = 0.1
X = (X / 255.0) - 0.5
count = len(X)

# ну и побольше итераций
for j in range(0, 30):
    print(j) # для удобства

    l1 = sigmoid(X.dot(W1) + b1)
    l2 = sigmoid(l1.dot(W2) + b2)

    # Нам нужно расчитать ошибки и дельты для каждого слоя
    l2_error = l2 - y  # просто вычтем из того что получилось то что нам нужно получить
    l2_delta = l2_error * sigmoid_prime(l2)  # я немного подредактировал фу-ю `sigmoid_prime`

    l1_error = l2_delta.dot(W2.T)
    l1_delta = l1_error * sigmoid_prime(l1)

    dW2 = l1.T.dot(l2_delta) / count  # это важный момент: нужно нормализовать значение разделив
    # на кол-во элементов в тренировочной выборке
    db2 = np.mean(l2_delta, axis=0)

    # и делаем тоже самое для первого слоя
    dW1 = X.T.dot(l1_delta) / count
    db1 = np.mean(l1_delta, axis=0)

    # теперь обновим наши веса (параметры)
    W1 += - alpha * dW1
    W2 += - alpha * dW2
    b1 += - alpha * db1
    b2 += - alpha * db2

"""
    Точность все равно около 20%
    Для того чтобы улучшить ее до 80-90% нужно сделать обучение mini-батчами позже я код скину, 
    а сейчас попробуйте сделать это сами. 
    
    Что нужно сделать:
    Обучение нужно проводить не на всем Х а на малой его части (например на 200-х картинках)
    
    X[0:200] 
    
    т.е. нужно сделать еще один внутренний цикл 
    в котором нужно делать Forward и Backward на кусочках от X  (X[0:200], X[200:400], X[400:600] и т.д.)
"""

print('Точность: ', calc_accuracy(orig_y, l2) * 100, '%')
