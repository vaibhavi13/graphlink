import networkx as nx
from gensim.models import Word2Vec
import random

def load_graph():
    graphfile = 'facebook_combined.txt'
    #graphfile = 'cora.txt'
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
    kwargs["sg"] = 1  # skip gram
    kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
    kwargs["workers"] = workers
    kwargs["window"] = window_size
    kwargs["epochs"] = iter

    print("Learning embedding vectors...")
    model = Word2Vec(**kwargs)
    print("Learning embedding vectors done!")
    print("model is \n")
    print(model)
    embeddings = {}
    for word in G.nodes():
        embeddings[word] = model.wv[word]
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
