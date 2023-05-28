import numpy as np
import matplotlib.pyplot as plt

import datagen

def y(x, w, b):
    return -(w[0]*x+b)/w[1]

def svm(data, labels, lamb, lr, num_iters):
    w = np.random.rand(2)
    b = np.random.rand()
    losses = []
    for _ in range(num_iters):
        loss = 0
        for i in range(data.shape[0]):
            correctness = labels[i] * (np.dot(w, data[i]) + b)
            loss += max(0, 1 - correctness)
            w -= lamb * w * lr
            if correctness < 1:
                w += labels[i] * data[i] * lr
                b += labels[i] * lr
        loss /= data.shape[0]
        loss += lamb * np.dot(w, w)
        losses.append(loss)
    return w, b, np.array(losses)

if __name__ == '__main__':
    cluster_size = 50
    data = datagen.gen_clusters(2, [-1, 1], [-1, 1], cluster_size, 0.1)
    labels = np.append(np.ones(cluster_size), -np.ones(cluster_size))

    x = np.linspace(-1, 1, 100)
    w, b, loss = svm(data, labels, 0.001, 0.001, 1000)

    plt.scatter(data[:cluster_size].T[0], data[:cluster_size].T[1], c='c')
    plt.scatter(data[cluster_size:].T[0], data[cluster_size:].T[1], c='m')
    plt.plot(x, y(x, w, b))
    plt.plot(x, y(x, w, b) + 1/w[1], '--')
    plt.plot(x, y(x, w, b) - 1/w[1], '--')
    plt.show()
    plt.plot(loss)
    plt.show()

