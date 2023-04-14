from gmaze import cluster
import numpy as np

points = np.array([ [9.0, 10.0], [1000, 9000],[1,2],[2,1],[2,2],[9,9],[1000,8000]])
k = 3
max_iters = 10
print(cluster.kmeans(points, k, max_iters))
