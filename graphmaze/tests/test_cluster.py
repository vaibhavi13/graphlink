from gmaze import cluster
import numpy as np

def test_kmeans():
    points = np.array([ [9.0, 10.0], [1000, 9000],[1,2],[2,1],[2,2],[9,9],[1000,8000]])
    k = 3
    max_iters = 10
    assert cluster.kmeans(points, k, max_iters) is not None

def test_kmeans_python():
    points = np.array([ [9.0, 10.0], [1000, 9000],[1,2],[2,1],[2,2],[9,9],[1000,8000]])
    k = 3
    max_iters = 10
    assert cluster.kmeans_python(points, k, max_iters) is not None