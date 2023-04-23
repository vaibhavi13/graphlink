import cupy as cp
from numba import cuda
import numpy as np
from pkg_resources import resource_filename

# CUDA kernel to assign each point to the nearest centroid using Python
@cuda.jit
def kmeans_kernel(points, centroids, cluster_assignments):
    idx = cuda.grid(1)
    if idx < points.shape[0]:
        min_dist = 10000000000
        min_idx = -1
        for i in range(centroids.shape[0]):
            dist = 0.0
            for j in range(centroids.shape[1]):
                diff = points[idx, j] - centroids[i, j]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        cluster_assignments[idx] = min_idx

class Kmeans:

    def __init__(self):

        #Code to call cuda kernel written in C++
        kernel_file_path = resource_filename("gmaze", "kernel.cu")
        with open(kernel_file_path, 'r') as file:
            source = file.read()
            module = cp.RawModule(code=source,backend='nvcc') #

        self.kernel = module.get_function('kmeans_kernel')

    def kmeans_cuda(self, points, k, max_iters):

        # Initialize centroids randomly
        centroids = cp.random.rand(k, points.shape[1]).astype(np.float32)
        points_gpu = cp.asarray(points).astype(np.float32)
        
        for iter in range(max_iters):
            # Assign points to nearest centroid using CUDA kernel
            cluster_assignments = cp.zeros(points.shape[0], dtype=np.int32)
            threads_per_block = 1024
            blocks_per_grid = (points.shape[0] + threads_per_block - 1) // threads_per_block
            
            
            self.kernel((blocks_per_grid,), (threads_per_block,), (points_gpu, centroids, cluster_assignments,k,points.shape[1],points.shape[0]))

            # Compute new centroids
            for i in range(k):
                mask = (cluster_assignments == i)
                if cp.any(mask):
                    centroids[i] = cp.mean(points_gpu[mask], axis=0)
            
        return cluster_assignments


    def kmeans_python(self, points, k, max_iters):

        # Initialize centroids randomly
        centroids = cp.random.rand(k, points.shape[1]).astype(np.float32)
        points_gpu = cp.asarray(points).astype(np.float32)

        for iter in range(max_iters):
            # Assign points to nearest centroid using CUDA kernel
            cluster_assignments = cp.zeros(points.shape[0], dtype=np.int32)
            threads_per_block = 1024
            blocks_per_grid = (points.shape[0] + threads_per_block - 1) // threads_per_block
            kmeans_kernel[blocks_per_grid, threads_per_block](points_gpu, centroids, cluster_assignments)

            # Compute new centroids
            for i in range(k):
                mask = (cluster_assignments == i)
                if cp.any(mask):
                    centroids[i] = cp.mean(points_gpu[mask], axis=0)

        return cluster_assignments



"""
Usage Example given below

# Using CUDA with C++
cluster_assignments = kmeans(points, k, max_iters)

# Using CUDA with Python
# cluster_assignments = kmeans_python(points, k, max_iters)

print(cluster_assignments)

"""

