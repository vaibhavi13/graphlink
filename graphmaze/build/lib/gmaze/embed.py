import gensim
import torch
import networkx as nx
from gensim.models import Word2Vec
import numpy as np

def word2vec(G, walks, embed_size=128, window_size=5, workers=-1, iter=5, **kwargs):
    kwargs["sentences"] = walks
    kwargs["min_count"] = kwargs.get("min_count", 0)
    kwargs["vector_size"] = embed_size
    kwargs["sg"] = 0  # BOW
    kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
    kwargs["window"] = window_size
    kwargs["epochs"] = iter
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs["compute_loss"] = True
    kwargs["callbacks"] = [gensim.models.callbacks.CallbackAny2Vec()]

    if device.type == 'cuda':
        # print('The model ran on the GPU')
        kwargs["compute_loss"] = False
    else:
        # print('The model ran on the CPU')
        kwargs["callbacks"] = None
        kwargs["workers"] = max(3,workers)

    print("Learning embedding vectors...")
    model = gensim.models.Word2Vec(**kwargs)
    print("Learning embedding vectors done!")
    embeddings = np.zeros((G.number_of_nodes(), embed_size))
    for i, node in enumerate(G.nodes()):
        embeddings[i] = model.wv[node]

    return embeddings