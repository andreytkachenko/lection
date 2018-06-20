
Загрузка датасета
-----------------
команда:
```bash
python mnist.py
```
результат:
```
Downloading train-images-idx3-ubyte.gz...
Downloading t10k-images-idx3-ubyte.gz...
Downloading train-labels-idx1-ubyte.gz...
Downloading t10k-labels-idx1-ubyte.gz...
Download complete.
Save complete.
```

Запуск
------
команда:
```bash
python main.py
```
результат (пример):
```
X:   (60000, 784)
y:   (60000, 10)
W1:  (784, 32)
b1:  (32,)
W2:  (32, 10)
b2:  (10,)
Точность:  10.165000000000003 %
```