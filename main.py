from gmaze import cluster
import numpy as np
from utils import load_graph, get_embedding, deepwalk_walks
import community as comm
from sklearn.cluster import KMeans
import time

G = load_graph()
l = 1
walks_deepwalk = deepwalk_walks(G, walk_length=l, num_walks=80)
embeddings_deepwalk = get_embedding(G,walks_deepwalk)
G1 = G.to_undirected()

points = [[0] for i in range(G1.number_of_nodes()-1)]
for i in range(0, G1.number_of_nodes()-1):
    points[i] = embeddings_deepwalk[str(i+1)]
points = np.array(points)

print("shape of points is "+ str(points.shape))

#modularity_scores = cluster_eval(G1, embeddings_deepwalk)
#points = np.array([ [9.0, 10.0], [1000, 9000],[1,2],[2,1],[2,2],[9,9],[1000,8000]])
#k = 3

max_iters = 10
NOC = 100

start_time = time.time()
print("\nStarting kmeans...")

for k in range(99,NOC):
    clusters = cluster.kmeans(points, k, max_iters)

print("\nFinished kmeans...")
end_time = time.time()

elapsed_time = end_time - start_time
print("Execution time:", elapsed_time, "seconds")
