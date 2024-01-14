import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def calculate_mean(points):
    return np.mean(points, axis=0)

def kmeans_clustering(labes_set, K, max_iter=100):
    num_points, num_features = labes_set.shape
    # Initialize cluster path_centers randomly
    path_centers = labes_set[np.random.choice(num_points, K, replace=False)]
    # Initialize path array
    path = np.full(num_points, -1)
  
    for iteration in range(max_iter):
        for i, point in enumerate(labes_set):
            min_distance = float('inf')
            for k, center in enumerate(path_centers):
                distance = euclidean_distance(point, center)
                if distance < min_distance:
                    min_distance = distance
                    path[i] = k
  
        new_path_centers = np.zeros((K, num_features))
        for k in range(K):
            cluster_points = labes_set[path == k]
            new_path_centers[k] = calculate_mean(cluster_points)
  
        if np.allclose(path_centers, new_path_centers):
            break
        else:
            path_centers = new_path_centers
  
    return path_centers, path