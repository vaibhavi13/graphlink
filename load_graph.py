import networkx as nx

graphfile = 'facebook_combined.txt'
# labelfile = 'facebook_combined.nodes.labels'
G = nx.read_edgelist('facebook_combined.txt', nodetype=None)
G = G.to_directed()
print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())
