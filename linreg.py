import numpy as np
import matplotlib.pyplot as plt

import datagen

if __name__ == '__main__':
    data = datagen.linear(num_points=100, slope=0.5, intercept=-3, mean=7, spread=0.1)
    plt.scatter(data.T[0], data.T[1])
    plt.show()

