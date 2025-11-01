def floyd_warshall(graph):
    """
    Compute shortest paths and next-hop routing tables using the Floyd–Warshall algorithm.

    Args:
        graph (dict): A dictionary-of-dictionaries representing the weighted directed graph.
                      Example: {u: {v: weight, ...}, ...}

    Returns:
        dist (dict): dist[u][v] gives the shortest distance from u to v.
        next_hop (dict): next_hop[u][v] gives the next node to take from u to reach v along the shortest path.
    """
    # Get all nodes in the graph
    nodes = set(graph.keys())
    for u in graph:
        nodes.update(graph[u].keys())
    nodes = list(nodes)

    # Initialize distance and next_hop tables
    dist = {u: {v: float('inf') for v in nodes} for u in nodes}
    next_hop = {u: {v: None for v in nodes} for u in nodes}
    for u in nodes:
        dist[u][u] = 0
        next_hop[u][u] = u
    for u in graph:
        for v in graph[u]:
            dist[u][v] = graph[u][v]
            next_hop[u][v] = v

    # Floyd–Warshall main loop
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_hop[i][j] = next_hop[i][k]

    return dist, next_hop

# Example usage:
# graph = {
#     "A": {"B": 5, "C": 10},
#     "B": {"C": 3},
#     "C": {"A": 1}
# }
# dist, next_hop = floyd_warshall(graph)
# print("Shortest distance from A to C:", dist["A"]["C"])
# print("Next hop from A to C:", next_hop["A"]["C"])