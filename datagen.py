import numpy as np

def gen_clusters(num_clusters, x_range, y_range, cluster_size, spread):
    data = np.array([])
    for _ in range(num_clusters):
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        mean = np.array([x, y])
        cluster = np.random.randn(cluster_size, 2) * spread + mean
        data = cluster if data.size == 0 else np.append(data, cluster, axis=0)
    return data

