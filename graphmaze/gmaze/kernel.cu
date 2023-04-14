#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

extern "C" __global__ void kmeans_kernel(float* points, float* centroids, int* cluster_assignments, int k, int d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float min_dist = 10000000000;
        int min_idx = -1;
        for (int i = 0; i < k; i++) {
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = points[idx * d + j] - centroids[i * d + j];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = i;
            }
        }
        cluster_assignments[idx] = min_idx;
    }
}


