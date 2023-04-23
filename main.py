from gmaze import cluster, embed
import numpy as np
from utils import load_graph, get_embedding, deepwalk_walks
import community as comm
from sklearn.cluster import KMeans
import time
import numpy as np



length = 10
max_iters = 10
k=10
num_walks=80
embed_size=128

kmeans = cluster.Kmeans()
G = load_graph(graphfile = 'facebook_combined.txt')
walks_deepwalk = deepwalk_walks(G, walk_length=length, num_walks=num_walks)


print("\nStarting kmeans on gpu...")
for _ in range(3):
    embeddings_deepwalk = embed.word2vec(G,walks_deepwalk,embed_size=embed_size)
    start_time = time.time()
    clusters = kmeans.kmeans_cuda(embeddings_deepwalk, k, max_iters)
    end_time = time.time()
    print("Finished kmeans...")
    elapsed_time = end_time - start_time
    print("Execution time:", elapsed_time, "seconds")


print("\nStarting kmeans on cpu...")
embeddings_deepwalk = get_embedding(G,walks_deepwalk,embed_size=embed_size)
start_time = time.time()
clusters = KMeans(n_clusters=k, random_state=0,max_iter=max_iters).fit(embeddings_deepwalk)
end_time = time.time()
print("Finished kmeans...")
elapsed_time = end_time - start_time
print("Execution time:", elapsed_time, "seconds")
