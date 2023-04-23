from gmaze import cluster
import numpy as np

kmeans = cluster.Kmeans()

def test_kmeans():
    points = np.array([ [9.0, 10.0], [1000, 9000],[1,2],[2,1],[2,2],[9,9],[1000,8000]])
    k = 3
    max_iters = 10
    assert kmeans.kmeans_cuda(points, k, max_iters) is not None

def test_kmeans_python():
    points = np.array([ [9.0, 10.0], [1000, 9000],[1,2],[2,1],[2,2],[9,9],[1000,8000]])
    k = 3
    max_iters = 10
    assert kmeans.kmeans_python(points, k, max_iters) is not None