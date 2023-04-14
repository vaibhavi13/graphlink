from gmaze import cluster
import numpy as np
from utils import load_graph, get_embedding, deepwalk_walks





# walks_deepwalk = deepwalk_walks(G, walk_length=l, num_walks=80)
# embeddings_deepwalk = get_embedding(G,walks_deepwalk)
# modularity_scores = cluster_eval(G1, embeddings_deepwalk)

points = np.array([ [9.0, 10.0], [1000, 9000],[1,2],[2,1],[2,2],[9,9],[1000,8000]])
k = 3
max_iters = 10
print(cluster.kmeans(points, k, max_iters))
