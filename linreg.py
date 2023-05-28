import numpy as np
import matplotlib.pyplot as plt

import datagen

def linreg(data):
    x_mean = np.mean(data.T[0])
    y_mean = np.mean(data.T[1])
    rx = data.T[0] - x_mean
    ry = data.T[1] - y_mean
    slope = np.dot(rx, ry) / np.dot(rx, rx)
    intercept = y_mean - slope * x_mean
    return slope, intercept

if __name__ == '__main__':
    mean = np.random.randn() * 2 - 1
    slope = np.random.randn() * 2 - 1
    intercept = np.random.randn() * 2 - 1

    data = datagen.linear(20, slope, intercept, mean, 1)

    a, b = linreg(data)
    x = np.linspace(mean-5, mean+5, 100)

    plt.xlim(mean-10, mean+10)
    plt.ylim(mean*slope - 10, mean*slope + 10)
    plt.scatter(data.T[0], data.T[1])
    plt.plot(x, a*x+b, c='r')
    plt.show()

