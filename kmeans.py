import numpy as np
import matplotlib.pyplot as plt

import datagen

def forgy(data, k):
    return data[np.random.choice(data.shape[0], k, replace=False), :]

def rand_part(data, k):
    groups = []
    for _ in range(k):
        groups.append([])
    for p in data:
        groups[np.random.randint(k)].append(p)
    centroids = np.array([np.mean(group, axis=0) for group in groups])
    return centroids

def kpp(data, k):
    index = np.random.randint(data.shape[0])
    centroids = data[index].reshape(1, 2)
    data = np.delete(data, index, axis=0)
    for _ in range(k-1):
        weights = np.array([])
        for p in data:
            vecs = centroids - p
            dists = [np.dot(v, v) for v in vecs]
            weights = np.append(weights, min(dists))
        index = np.random.choice(np.arange(data.shape[0]), 1, p=weights/sum(weights))
        centroids = np.append(centroids, data[index], axis=0)
    return centroids

def kmeans(data, k, num_iters, animate=False):
    centroids = kpp(data, k)
    if animate:
        plt.scatter(data.T[0], data.T[1], color='k')
        for centroid in centroids:
            plt.scatter(centroid[0], centroid[1], marker='X', edgecolors='k')
        plt.draw()
        plt.pause(1)
        plt.clf()
    for _ in range(num_iters):
        groups = []
        for _ in range(k):
            groups.append([])
        for p in data:
            vecs = centroids - p
            dists = [np.dot(v, v) for v in vecs]
            min_group = min(enumerate(dists), key=lambda x: x[1])[0]
            groups[min_group].append(p)
        for i in range(k):
            if len(groups[i]) != 0:
                centroids[i] = np.mean(groups[i], axis=0)
            if animate:
                cent_plot = plt.scatter(centroids[i][0], centroids[i][1], marker='X', edgecolors='k')
                color = cent_plot.get_facecolors()
                if len(groups[i]) != 0:
                    plt.scatter(np.array(groups[i]).T[0], np.array(groups[i]).T[1], color=color, zorder=-1)
        plt.draw()
        plt.pause(0.5)
        plt.clf()
    return centroids

if __name__ == '__main__':
    k = 7
    data = datagen.clusters(num_clusters=k, x_range=[0, 10], y_range=[0, 10], cluster_size=50, spread=0.3)
    centroids = kmeans(data, k, 20, animate=True)

