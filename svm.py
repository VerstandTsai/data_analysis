import numpy as np
import matplotlib.pyplot as plt

import datagen

if __name__ == '__main__':
    data = datagen.gen_clusters(2, [0, 10], [0, 10], 50, 0.5)
    plt.scatter(data.T[0], data.T[1])
    plt.show()

