import networkx as nx
from gensim.models import Word2Vec
import random
import numpy as np
import os
import torch



def load_graph(graphfile = 'cora.txt'):
    print("data file used is "+ graphfile)
    # labelfile = 'facebook_combined.nodes.labels'
    G = nx.read_edgelist(graphfile, nodetype=None)
    G = G.to_directed()
    print("Number of nodes: ", G.number_of_nodes())
    print("Number of edges: ", G.number_of_edges())
    return G

def get_embedding(G, walks, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
    kwargs["sentences"] = walks
    kwargs["min_count"] = kwargs.get("min_count", 0)
    kwargs["vector_size"] = embed_size
    kwargs["sg"] = 0  # skip gram
    kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
    kwargs["workers"] = workers
    kwargs["window"] = window_size
    kwargs["epochs"] = iter

    print("Learning embedding vectors...")
    model = Word2Vec(**kwargs)

    print("Learning embedding vectors done!")
    embeddings = np.zeros((G.number_of_nodes(), embed_size))
    for i, node in enumerate(G.nodes()):
        embeddings[i] = model.wv[node]
    return embeddings





def deepwalk_walks(G, num_walks, walk_length,):
        nodes = G.nodes()
        walks = []
        for _ in range(num_walks):
            for v in nodes:
                walk = [v]
                while len(walk) < walk_length:
                    cur = walk[-1]
                    cur_nbrs = list(G.neighbors(cur))
                    if len(cur_nbrs) > 0:
                        walk.append(random.choice(cur_nbrs))
                    else:
                        break
                walks.append(walk)
        return walks
