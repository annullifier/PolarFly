# PolarFly Topology Visualization and Analysis Tool
# Author: Christian Martin
# November, 2024
# ==================================================

# Purpose:
# --------
# This program implements and analyzes the PolarFly network topology, a novel high-performance
# interconnection network based on finite projective geometry. PolarFly is designed for
# high-performance computing systems, data centers, and large-scale distributed systems.

# PolarFly Topology Overview:
# -------------------------
# PolarFly is a family of network topologies derived from polar spaces in finite projective
# geometry. For a given prime number q, it generates a network with the following properties:

# - Node count: q² + q + 1 (k² - k + 1, where k = q + 1)
# - Regular degree: q + 1
# - Diameter: 2 (maximum distance between any two nodes)
# - Multiple parallel paths between node pairs
# - Near-optimal fault tolerance
# - Balanced traffic distribution

# Key Benefits:
# ------------
# 1. Scalability:
#    - Efficient scaling with increasing q
#    - Maintains constant diameter (2) regardless of size
#    - Better cost/performance ratio compared to traditional topologies

# 2. Performance:
#    - Low latency due to 2-hop maximum path length
#    - High bandwidth through multiple parallel paths
#    - Excellent congestion resistance

# 3. Reliability:
#    - High fault tolerance through path diversity
#    - Graceful degradation under failures
#    - Maintains connectivity even with multiple failures

# 4. Implementation:
#    - Regular structure simplifies routing
#    - Symmetric properties enable efficient load balancing
#    - Suitable for both physical and logical network implementations

# Mathematical Foundation:
# ----------------------
# The topology is constructed using properties of finite projective planes:
# - Based on finite field GF(q) where q is prime
# - Nodes represent 1-dimensional subspaces
# - Connections based on orthogonality relationships
# - Leverages projective geometry for optimal structural properties

# This Implementation:
# ------------------
# This code provides:
# 1. Core topology generation using finite field arithmetic
# 2. Interactive 3D visualization with path highlighting
# 3. Comprehensive network metrics analysis
# 4. Multiple output formats for further analysis
# 5. Tools for studying path diversity and fault tolerance

# Usage:
# ------
# The implementation supports:
# - Topology generation for any prime q
# - Interactive exploration of network properties
# - Analysis of path characteristics
# - Generation of connection lists and adjacency matrices
# - Visualization of network structure and paths

# Valid values of q through 256 are:
# 2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37, 41, 
# 43, 47, 49, 53, 59, 61, 64, 67, 71, 73, 79, 81, 83, 89, 97, 101, 103, 107, 
# 109, 113, 121, 125, 127, 128, 131, 137, 139, 149, 151, 157, 163, 167, 169, 
# 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 243, 
# 251, 256
# Note: Need to fix the case when q is a prime power!! Need new math for that
# stay tuned.

# Note: The choice of q determines the network size and should be selected
# based on the specific application requirements and scaling needs.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from heapq import heappush, heappop
from itertools import combinations

# Field Operations and Vector Space Generation
# ------------------------------------------
# These functions implement finite field arithmetic operations necessary for PolarFly topology
# The field operations are based on modular arithmetic for prime fields GF(q)

def compute_coeffs(i, primePower, primeFactor):
    # Converts an integer to its polynomial representation in the finite field
    # Uses a base-p representation where p is the prime factor
    coeffs = [0 for j in range(primePower)]
    tmp = int(i)
    cid = 0
    while(tmp > 0):
        coeffs[cid] = tmp % primeFactor
        cid += 1
        tmp = int(tmp/primeFactor)
    return coeffs

def compute_index(coeffs, primePower, primeFactor):
    cid = primePower-1
    index = 0
    while(cid >= 0):
        index = index*primeFactor + coeffs[cid]
        cid -= 1 
    return index

def field_gen(q):
    add_mat = [[0]*q for i in range(q)]
    mul_mat = [[0]*q for i in range(q)]
    
    # For prime field GF(q), simple modular arithmetic
    for i in range(q):
        for j in range(q):
            add_mat[i][j] = (i + j) % q
            mul_mat[i][j] = (i * j) % q
            
    return add_mat, mul_mat

class BrownTopology:
    def __init__(self, q):
        self.q = q
        self.add_mat, self.mul_mat = field_gen(q)
        
    def vector_mul(self, point, a):
        out = np.zeros(3, dtype=int)
        for i in range(3):
            out[i] = self.mul_mat[a][point[i]]
        return out
    
    def vec_dp(self, v1, v2):
        dp = 0
        for i in range(3):
            prod = self.mul_mat[v1[i]][v2[i]]
            dp = self.add_mat[dp][prod]
        return dp
    
    def generate(self):
        V = self.q**2 + self.q + 1
        vectors = []
        node_map = {}
        graph = [[] for _ in range(V)]
        
        # Generate 1-d subspace vectors
        for d1 in range(self.q):
            for d2 in range(self.q):
                v = (d1, d2, 1)
                vectors.append(v)
        
        for d1 in range(self.q):
            v = (d1, 1, 0)
            vectors.append(v)
            
        vectors.append((1, 0, 0))
        
        # Create node mapping
        for idx, v in enumerate(vectors):
            node_map[v] = idx
            
        # Create connections
        for idx, v in enumerate(vectors):
            for vv in vectors:
                dp = self.vec_dp(v, vv)
                if (dp % self.q) == 0:
                    source = idx
                    dest = node_map[vv]
                    if dest != source:
                        graph[source].append(dest)
        
        return graph

def find_all_shortest_paths(G, source, target):
    """Find all shortest paths between source and target"""
    return list(nx.all_shortest_paths(G, source, target))

def find_second_shortest_paths(G, source, target):
    """Find all second shortest paths between source and target"""
    shortest_length = nx.shortest_path_length(G, source, target)
    all_paths = []
    
    # Use a modified BFS to find all paths up to length shortest_length + 3
    def bfs_paths(G, source, target, cutoff):
        queue = [(source, [source])]
        while queue:
            (vertex, path) = queue.pop(0)
            for next in G[vertex]:
                if next == target and len(path) <= cutoff:
                    yield path + [next]
                elif next not in path and len(path) < cutoff:
                    queue.append((next, path + [next]))
    
    # Get all paths up to length shortest_length + 3
    max_length = min(shortest_length + 3, len(G.nodes) - 1)  # Don't exceed n-1 hops
    for path in bfs_paths(G, source, target, max_length):
        if len(path) - 1 > shortest_length:  # -1 because path includes both endpoints
            all_paths.append(path)
    
    # Filter to keep only the shortest among the longer paths
    if all_paths:
        min_length = min(len(path) for path in all_paths)
        second_shortest = [path for path in all_paths if len(path) == min_length]
        return second_shortest
    return []


def visualize_brown_topology_3d_polar(graph, q):
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes and edges
    for i in range(len(graph)):
        G.add_node(i)
    for i, neighbors in enumerate(graph):
        for neighbor in neighbors:
            G.add_edge(i, neighbor)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(111, projection='3d')
    
    # Calculate node positions (cylindrical layout)
    def cylindrical_layout(G):
        pos = {}
        nodes = list(G.nodes())
        n = len(nodes)
        layers = int(np.sqrt(n))
        nodes_per_layer = n // layers + 1
        
        for i, node in enumerate(nodes):
            layer = i // nodes_per_layer
            pos_in_layer = i % nodes_per_layer
            theta = (2 * np.pi * pos_in_layer) / nodes_per_layer
            r = 1.0
            h = 2.0 * layer / layers - 1.0
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = h
            pos[node] = np.array([x, y, z])
        return pos
    
    pos_cylindrical = cylindrical_layout(G)
    
    # Extract coordinates
    node_xyz = np.array([pos_cylindrical[v] for v in G.nodes()])
    edge_xyz = np.array([(pos_cylindrical[u], pos_cylindrical[v]) for u, v in G.edges()])
    
    # Create empty lists for edge lines and selected nodes
    edge_lines = []
    selected_nodes = []
    
    # Plot edges
    for edge in edge_xyz:
        line = ax1.plot3D(*edge.T, color='gray', alpha=0.3, linewidth=0.8)[0]
        edge_lines.append(line)
    
    # Plot nodes
    scatter_points = ax1.scatter(node_xyz[:,0], node_xyz[:,1], node_xyz[:,2],
                               c='lightblue', s=200, edgecolors='darkblue',
                               alpha=0.8, picker=True)
    
    # Add node labels
    for i, (x, y, z) in enumerate(node_xyz):
        ax1.text(x*1.1, y*1.1, z*1.1, str(i), fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # def reset_colors():
    #     """Reset all colors to default"""
    #     scatter_points.set_facecolor('lightblue')
    #     scatter_points.set_edgecolor('darkblue')
    #     for line in edge_lines:
    #         line.set_color('gray')
    #         line.set_alpha(0.3)
    #         line.set_linewidth(0.8)

    # def highlight_path(path, color, width=2.0, alpha=1.0):
    #     """Highlight a path with specified color"""
    #     for i in range(len(path)-1):
    #         u, v = path[i], path[i+1]
    #         edge_found = False
    #         for idx, (s, t) in enumerate(G.edges()):
    #             if (s == u and t == v) or (s == v and t == u):
    #                 edge_lines[idx].set_color(color)
    #                 edge_lines[idx].set_alpha(alpha)
    #                 edge_lines[idx].set_linewidth(width)
    #                 edge_found = True
    #                 break

    def reset_colors():
        """Reset all colors to default"""
        scatter_points.set_facecolor('lightblue')
        scatter_points.set_edgecolor('darkblue')
        for line in edge_lines:
            line.set_color('gray')
            line.set_alpha(0.3)
            line.set_linewidth(0.8)

    def highlight_path(path, color, width=2.0, alpha=1.0):
        """Highlight a path with specified color"""
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            for idx, (s, t) in enumerate(G.edges()):
                if (s == u and t == v) or (s == v and t == u):
                    edge_lines[idx].set_color(color)
                    edge_lines[idx].set_alpha(alpha)
                    edge_lines[idx].set_linewidth(width)
                    break

    # def on_pick(event):
    #     ind = event.ind[0]
    #     node = list(G.nodes())[ind]
        
    #     if node not in selected_nodes:
    #         if len(selected_nodes) == 0:
    #             # First node selected
    #             reset_colors()
    #             selected_nodes.append(node)
    #             colors = scatter_points.get_facecolor()
    #             colors[ind] = np.array([1, 1, 0, 0.8])  # Yellow for first selected node
    #             scatter_points.set_facecolor(colors)
            
    #         elif len(selected_nodes) == 1:
    #             # Second node selected
    #             selected_nodes.append(node)
    #             colors = scatter_points.get_facecolor()
    #             colors[ind] = np.array([1, 1, 0, 0.8])  # Yellow for second selected node
    #             scatter_points.set_facecolor(colors)
                
    #             # Find and highlight paths
    #             shortest_paths = find_all_shortest_paths(G, selected_nodes[0], selected_nodes[1])
    #             second_shortest = find_second_shortest_paths(G, selected_nodes[0], selected_nodes[1])
                
    #             # Highlight shortest paths in green
    #             for path in shortest_paths:
    #                 highlight_path(path, 'green')
                
    #             # Highlight second shortest paths in red
    #             for path in second_shortest:
    #                 highlight_path(path, 'red')
                
    #         else:
    #             # Reset for new selection
    #             reset_colors()
    #             selected_nodes.clear()
    #             selected_nodes.append(node)
    #             colors = scatter_points.get_facecolor()
    #             colors[ind] = np.array([1, 1, 0, 0.8])  # Yellow for first selected node
    #             scatter_points.set_facecolor(colors)
        
    #     fig.canvas.draw_idle()

    def on_pick(event):
        # Only handle pick events (node clicks), not canvas clicks
        if not hasattr(event, 'ind'):
            return
            
        ind = event.ind[0]
        node = list(G.nodes())[ind]
        print(f"Clicked node {node} at index {ind}")
        
        if len(selected_nodes) == 0:
            # First node selected
            reset_colors()
            selected_nodes.append(node)
            colors = scatter_points.get_facecolor()
            colors[ind] = np.array([1, 1, 0, 0.8])  # Yellow
            scatter_points.set_facecolor(colors)
            
        elif len(selected_nodes) == 1:
            # Second node selected
            if node != selected_nodes[0]:
                selected_nodes.append(node)
                first_ind = list(G.nodes()).index(selected_nodes[0])
                
                # Reset all colors first
                reset_colors()
                
                # Only set the two selected nodes to yellow
                colors = scatter_points.get_facecolor()
                colors[first_ind] = np.array([1, 1, 0, 0.8])  # First node yellow
                colors[ind] = np.array([1, 1, 0, 0.8])  # Second node yellow
                scatter_points.set_facecolor(colors)
                
                # Find and highlight paths
                shortest_paths = find_all_shortest_paths(G, selected_nodes[0], selected_nodes[1])
                second_shortest = find_second_shortest_paths(G, selected_nodes[0], selected_nodes[1])
                
                # Highlight paths
                for path in shortest_paths:
                    highlight_path(path, 'green')
                for path in second_shortest:
                    highlight_path(path, 'red')
        
        else:
            # More than 2 nodes, reset
            reset_colors()
            selected_nodes.clear()
            selected_nodes.append(node)
            colors = scatter_points.get_facecolor()
            colors[ind] = np.array([1, 1, 0, 0.8])  # Yellow
            scatter_points.set_facecolor(colors)
        
        fig.canvas.draw_idle()

    # Only connect pick event
    fig.canvas.mpl_connect('pick_event', on_pick)
    # fig.canvas.mpl_connect('button_press_event', on_pick)
    
    # Add rotation controls
    def rotate(event):
        if event.key == 'left':
            ax1.view_init(azim=ax1.azim + 10)
        elif event.key == 'right':
            ax1.view_init(azim=ax1.azim - 10)
        elif event.key == 'up':
            ax1.view_init(elev=ax1.elev + 10)
        elif event.key == 'down':
            ax1.view_init(elev=ax1.elev - 10)
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', rotate)
    
    # Add graph metrics
    metrics = f"""
    Nodes: {G.number_of_nodes()}
    Edges: {G.number_of_edges()}
    Links per Node: {(2 * G.number_of_edges() / G.number_of_nodes()):.1f}
    Min Degree: {min(dict(G.degree()).values())}
    Max Degree: {max(dict(G.degree()).values())}
    Density: {nx.density(G):.3f}
    Diameter: {nx.diameter(G)}
    Avg Path Length: {nx.average_shortest_path_length(G):.2f}
    Clustering Coeff: {nx.average_clustering(G):.3f}
    
    Click nodes to show paths:
    - Shortest paths: Green
    - Second shortest: Red
    """
    fig.text(0.02, 0.02, metrics, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    ax1.set_title('Cylindrical Layout', fontsize=14)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.grid(True, alpha=0.2)
    ax1.view_init(elev=20, azim=45)
    
    plt.suptitle(f'PolarFly Topology (q={q}) - 3D Polar Layout\n', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return G

def analyze_topology_metrics(G):
    print("\nDetailed Network Analysis:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"Density: {nx.density(G):.3f}")
    print(f"Diameter: {nx.diameter(G)}")
    print(f"Average shortest path length: {nx.average_shortest_path_length(G):.3f}")
    print(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    print(f"\nDegree distribution:")
    print(f"Min degree: {min(degrees)}")
    print(f"Max degree: {max(degrees)}")
    print(f"Mean degree: {sum(degrees)/len(degrees):.2f}")

def generate_connection_list(graph):
    """
    Generates a list of tuples representing all connections in the graph.
    Each tuple contains (source_node, destination_node).
    """
    connections = []
    for source, destinations in enumerate(graph):
        for dest in destinations:
            # Only add each edge once (avoiding duplicates)
            if source < dest:
                connections.append((source, dest))
    return connections

def generate_adjacency_matrix(graph):
    """
    Generates a symmetric adjacency matrix from the graph.
    1 indicates a connection, 0 indicates no connection.
    """
    n = len(graph)
    adj_matrix = np.zeros((n, n), dtype=int)
    
    for source, destinations in enumerate(graph):
        for dest in destinations:
            adj_matrix[source][dest] = 1
            adj_matrix[dest][source] = 1  # Symmetric matrix for undirected graph
            
    return adj_matrix

def save_matrix_outputs(adj_matrix, q):
    """
    Saves the adjacency matrix to:
    1. Text file (pf_adj.txt)
    2. LaTeX code (pf_adj_latex.tex)
    """
    # Save to text file
    with open('pf_adj.txt', 'w') as f:
        f.write(f"Adjacency Matrix for PolarFly (q={q}):\n\n")
        n = len(adj_matrix)
        # Write column headers
        f.write("    ") # Initial spacing for row labels
        for j in range(n):
            f.write(f"{j:4}")
        f.write("\n")
        # Write matrix with row labels
        for i, row in enumerate(adj_matrix):
            f.write(f"{i:4}")  # Row label
            for val in row:
                f.write(f"{val:4}")
            f.write("\n")
    
    # Generate and save LaTeX code
    with open('pf_adj_latex.tex', 'w') as f:
        f.write("% LaTeX code for PolarFly adjacency matrix\n")
        f.write(f"% q = {q}\n\n")
        f.write("\\begin{equation*}\n")
        f.write("A = \\begin{bmatrix}\n")
        for row in adj_matrix:
            f.write(" & ".join(map(str, row)) + " \\\\\n")
        f.write("\\end{bmatrix}\n")
        f.write("\\end{equation*}")


if __name__ == "__main__":
    q = 7
    brown = BrownTopology(q)
    topology = brown.generate()
    
    # Generate and display connection list
    connections = generate_connection_list(topology)
    print("\nConnection List:")
    for conn in connections:
        print(f"{conn[0]} -- {conn[1]}")
        
    # Generate and display adjacency matrix
    adj_matrix = generate_adjacency_matrix(topology)
    print("\nAdjacency Matrix:")
    print(adj_matrix)

    # Save matrix outputs
    save_matrix_outputs(adj_matrix, q)  # Add this line here
    
    # Create and visualize the graph
    G = visualize_brown_topology_3d_polar(topology, q)
    # Analyze metrics
    analyze_topology_metrics(G)
