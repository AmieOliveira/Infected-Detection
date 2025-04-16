import networkx as nx

def degree(graph):
    """
    Computes the number of neighbors of each node, normalized by n - 1.

    Parameters:
    - graph: NetworkX graph
    
    Returns:
    - A dictionary with, for each node, the fraction of nodes it is connected to.
    """
    
    return nx.degree_centrality(graph)

def contact(graph, observed_nodes):
    """
    Computes the fraction of observed infected neighbors of each node.

    Parameters:
    - graph: NetworkX graph
    - observed_nodes: List of observed infected nodes.
    
    Returns:
    - A dictionary with the fraction of observed infected neighbors for each node.
    """
    contact = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 0:
            contact[node] = 0.0
            continue
        num_infected_neighbors = sum(1 for neighbor in neighbors if neighbor in observed_nodes)
        contact[node] = num_infected_neighbors/len(neighbors)
    
    return contact

def contact_k(graph, observed_nodes, k):
    """
    Computes the fraction of observed infected nodes within distance k of each node in the graph.

    Parameters:
    - graph: NetworkX graph
    - observed_nodes: List of observed infected nodes.
    - k: Maximum distance to search from each node.
    
    Returns:
    - A dictionary with the fraction of observed infected nodes within distance k for each node.
    """
    contact_k = {}
    for node in graph.nodes():
        k_neighbors = list(nx.single_source_shortest_path_length(graph, node, cutoff=k).keys())
        if len(k_neighbors) == 0:
            contact_k[node] = 0.0
            continue
        num_infected_k_neighbors = sum(1 for v in k_neighbors if v in observed_nodes)
        contact_k[node] = num_infected_k_neighbors/len(k_neighbors)
    
    return contact_k

def contact_k_vector(graph, observed_nodes, k):
    """
    Computes the fraction of observed infected nodes at exactly distances from 1 to k of each node in the graph.

    Parameters:
    - graph: NetworkX graph
    - observed_nodes: List of observed infected nodes.
    - k: Maximum distance to search from each node.
    
    Returns:
    - A dictionary with the fraction of observed infected nodes at exactly distance from 1 to k for each node.
    """
    contact_k = {}
    for node in graph.nodes():
        neighbors = list(nx.single_source_shortest_path_length(graph, node, cutoff=k).keys())
        fractions = []
        for dist in range(1, k+1):
            dist_neighbors = [n for n, d in neighbors.items() if d == dist]
            if dist_neighbors:
                num_infected_k_neighbors = sum(1 for v in dist_neighbors if v in observed_nodes)
                fractions.append(num_infected_k_neighbors/len(dist_neighbors))
            else:
                fractions.append(0.0)
        contact_k[node] = fractions
    
    return contact_k

def betweenness(graph):
    """
    Computes betweenness centrality for all nodes.

    Parameters:
    - graph: NetworkX graph
    
    Returns:
    - A dictionary with the betweenness centrality scores.
    """

    return nx.betweenness_centrality(graph)

def observed_betweenness(graph, observed_nodes):
    """
    Computes the betweenness centrality of nodes considering only
    shortest paths that have both source and target nodes in the
    observed infected nodes set.

    Parameters:
    - graph: NetworkX graph
    - observed_nodes: List of observed infected nodes.

    Returns:
    - A dictionary with the betweenness centrality scores.
    """

    return nx.betweenness_centrality_subset(graph, sources=observed_nodes, targets=observed_nodes)