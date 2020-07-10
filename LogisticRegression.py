import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
seed = 123456
np.random.seed(seed)

def fun(x):
    return w*x+b

def compute_prob(w, b, x):
    z = x@w.transpose() + b
    return 1 / (1 + np.exp(-z))


def weight_descent(w, b, x, y, learning_rate):
    p = compute_prob(w, b, x)
    p = p.reshape(-1)
    dl_dz = - (y * (1 - p) / (np.log(p + 0.1) + 1) - (1 - y) * p / (np.log(1 - p + 0.1) + 1))
    dl_dw = np.mean(x * np.repeat(dl_dz, 2).reshape(dl_dz.shape[0], -1), axis=0)
    dl_db = np.mean(dl_dz)
    w -= dl_dw * learning_rate
    b -= dl_db * learning_rate
    return w, b

def compute_loss(w, b, x, y):
    p = compute_prob(w, b, x)
    p = p.reshape(-1)
    l =  - np.mean(y * np.log(p + 0.1) + (1 - y) * np.log(1 - p + 0.1))
    return l

def train(w, b, x, y, learning_rate):
    loss = compute_loss(w, b, x, y)
    w, b = weight_descent(w, b, x, y, learning_rate)
    return loss, w, b


if __name__=='__main__':
    w = np.random.randn(1,2)
    b = np.zeros(1)
    x, y = datasets.make_classification(n_samples=1000, n_classes=2, n_features=2, n_informative=2, n_redundant=0, n_repeated=0)
    x_train, x_test = x[:int(x.shape[0] * 0.8)], x[int(x.shape[0] * 0.8):]
    y_train, y_test = y[:int(y.shape[0] * 0.8)], y[int(y.shape[0] * 0.8):]
    epoch = 1000
    for i in range(epoch):
        l, w, b = train(w, b, x_train, y_train, 0.01)
        print("epoch:{} w1:{} w2:{} b:{} l:{}".format(i, w[0, 0], w[0, 1], b[0], l))
        prob = compute_prob(w, b, x_test) > 0.5
        pred = prob.astype(int).reshape(-1)
        correct = pred == y_test
        correct = correct.astype(int)
        correct = correct.sum()
        print("acc:{}".format(correct / y_test.shape[0]))

    w = float(w[0,0]/w[0, 1])
    b = float(b[0])
    plt.scatter(x_test[:, 0], x_test[:, 1], marker='*', c = y_test)
    x = np.linspace(-5, 5, 10).tolist()
    y = list(map(fun, x))
    plt.plot(x, y)
    plt.show()




