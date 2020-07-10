import numpy as np
from matplotlib import pyplot as plt

def generate_data():
    w = np.array([1.2])
    b = np.array([4.])
    x = np.linspace(-100, 100, 50)
    y = w * x + b
    error = np.random.randn(y.shape[0]) * 10
    return x,y

def gradient_descent(x, y, w, b, learning_rate):
    pred = x * w + b
    w_grad = np.mean((pred - y) * x) * 2
    b_grad = np.mean(pred - y) * 2
    w -= learning_rate * w_grad
    b -= learning_rate * b_grad
    return w, b

def compute_loss(x, y, w, b):
    pred = x * w + b
    return np.mean(np.power(pred - y, 2))

def train(x, y, w, b, learning_rate):
    loss = compute_loss(x, y, w, b)
    w, b = gradient_descent(x, y, w, b, learning_rate)
    return loss, w, b


if __name__=='__main__':
    x, y = generate_data()
    w = np.random.randn(1)
    b = np.random.rand(1)
    epoch = 10000
    for i in range(epoch):
        loss, w, b = train(x, y, w, b, 0.0001)
        print("epoch:{} w:{} b:{} loss:{}".format(i, w[0], b[0], float(loss)))
    l = np.linspace(-100, 100, 100)
    pred = l * w + b
    plt.scatter(x, y)
    plt.plot(l, pred)
    plt.show()



