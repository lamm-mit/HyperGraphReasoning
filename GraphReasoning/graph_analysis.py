# Analysis functions for graph data, including community detection and path finding
import os
import re
import math
import random
from copy import deepcopy
from datetime import datetime

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pyvis.network import Network
from sklearn.cluster import KMeans
import community as community_louvain


def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors."""
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def heuristic_path_with_embeddings(G, embedding_tokenizer, embedding_model, source, target, node_embeddings, top_k=3, second_hop=False, data_dir='./', save_files=True, verbatim=False):
    """
    Find a path between 'source' and 'target' by heuristic sampling using embedding similarity.
    """
    G = deepcopy(G)
    if verbatim:
        print("Original:", source, "-->", target)
    # Get best matching nodes for source and target keywords
    source = find_best_fitting_node_list(source, node_embeddings, embedding_tokenizer, embedding_model, 5)[0][0].strip()
    target = find_best_fitting_node_list(target, node_embeddings, embedding_tokenizer, embedding_model, 5)[0][0].strip()
    if verbatim:
        print("Selected:", source, "-->", target)

    def heuristic(node_current, node_target):
        """Estimate distance from current node to target using embeddings."""
        return euclidean_distance(node_embeddings[node_current], node_embeddings[node_target])

    def sample_path(node_current, visited):
        path = [node_current]
        while node_current != target:
            neighbors = [
                (nbr, heuristic(nbr, target)) for nbr in G.neighbors(node_current)
                if nbr not in visited
            ]
            if not neighbors:
                # Dead end, backtrack
                if len(path) > 1:
                    visited.add(path.pop())
                    node_current = path[-1]
                    continue
                else:
                    return None
            neighbors.sort(key=lambda x: x[1])
            top_neighbors = neighbors[:top_k] if len(neighbors) > top_k else neighbors
            next_node = random.choice(top_neighbors)[0]
            path.append(next_node)
            visited.add(next_node)
            node_current = next_node
            if len(path) > 2 * len(G):
                print(f"No path found between {source} and {target}")
                return None
        return path

    visited = {source}
    path = sample_path(source, visited)
    if path is None:
        print(f"No path found between {source} and {target}")
        return None, None, None, None, None

    # Include 2-hop neighbors if requested
    subgraph_nodes = set(path)
    if second_hop:
        for node in list(path):
            for neighbor in G.neighbors(node):
                subgraph_nodes.add(neighbor)
                for second_hop_neighbor in G.neighbors(neighbor):
                    subgraph_nodes.add(second_hop_neighbor)
    subgraph = G.subgraph(subgraph_nodes).copy()

    fname = None
    graph_GraphML = None
    if save_files:
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        nt = Network('500px', '1000px')
        nt.from_nx(subgraph)
        fname = f'{data_dir}/shortest_path_{time_tag}_{source}_{target}.html'
        nt.show(fname)
        if verbatim:
            print(f"HTML visualization: {fname}")
        graph_GraphML_name = f'shortestpath_{time_tag}_{source}_{target}.graphml'
        save_graph_without_text(subgraph, data_dir=data_dir, graph_name=graph_GraphML_name)
        graph_GraphML = graph_GraphML_name
        if verbatim:
            print(f"GraphML file: {graph_GraphML}")

    shortest_path_length = len(path) - 1
    return path, subgraph, shortest_path_length, fname, graph_GraphML

def find_shortest_path(G, source, target, verbatim=True, data_dir='./'):
    """Find and visualize shortest path between two nodes using NetworkX and PyVis."""
    path = nx.shortest_path(G, source=source, target=target)
    path_graph = G.subgraph(path)
    nt = Network('500px', '1000px')
    nt.from_nx(path_graph)
    fname = f'{data_dir}/shortest_path_{source}_{target}.html'
    nt.show(fname)
    if verbatim:
        print(f"Visualization: {fname}")
    graph_GraphML = f'{data_dir}/shortestpath_{source}_{target}.graphml'
    nx.write_graphml(path_graph, graph_GraphML)
    shortest_path_length = nx.shortest_path_length(G, source=source, target=target)
    return path, path_graph, shortest_path_length, fname, graph_GraphML

def find_N_paths(G, source, target, N=5, data_dir='./'):
    """Sample up to N simple paths between source and target and visualize each."""
    sampled_paths = []
    fname_list = []
    paths_generator = nx.all_simple_paths(G, source=source, target=target)
    for _ in range(N):
        try:
            path = next(paths_generator)
            sampled_paths.append(path)
        except StopIteration:
            break
    for i, path in enumerate(sampled_paths):
        nt = Network('500px', '1000px')
        path_graph = G.subgraph(path)
        nt.from_nx(path_graph)
        fname = f'{data_dir}/simple_path_{source}_{target}_{i}.html'
        nt.show(fname)
        print(f"Path {i+1} Visualization: {fname}")
        fname_list.append(fname)
    return sampled_paths, fname_list

def find_all_triplets(G):
    """Find all triplets of nodes in G that form a connected subgraph with exactly 3 edges."""
    from itertools import combinations
    triplets = []
    for nodes in combinations(G.nodes(), 3):
        subgraph = G.subgraph(nodes)
        if nx.is_connected(subgraph) and subgraph.number_of_edges() == 3:
            triplets.append(f"{nodes[0]}-{nodes[1]}-{nodes[2]}")
    return triplets

def print_node_pairs_edge_title(G):
    """
    Return a list of strings representing edges in the format: node_1, edge_title, node_2
    """
    pairs_and_titles = []
    for node1, node2, data in G.edges(data=True):
        title = data.get('title', 'No title')
        pairs_and_titles.append(f"{node1}, {title}, {node2}")
    return pairs_and_titles

def find_path(G, node_embeddings, tokenizer, model, keyword_1, keyword_2, verbatim=True, second_hop=False, data_dir='./', similarity_fit_ID_node_1=0, similarity_fit_ID_node_2=0, save_files=True):
    """
    Find a shortest path between two best-matching nodes for keyword_1 and keyword_2 in the embedding space.
    """
    # Get best matching nodes for the keywords
    best_node_1, best_similarity_1 = find_best_fitting_node_list(keyword_1, node_embeddings, tokenizer, model, max(5, similarity_fit_ID_node_1+1))[similarity_fit_ID_node_1]
    best_node_2, best_similarity_2 = find_best_fitting_node_list(keyword_2, node_embeddings, tokenizer, model, max(5, similarity_fit_ID_node_2+1))[similarity_fit_ID_node_2]
    if verbatim:
        print(f"{similarity_fit_ID_node_1}th best node for '{keyword_1}': '{best_node_1}' (sim={best_similarity_1})")
        print(f"{similarity_fit_ID_node_2}th best node for '{keyword_2}': '{best_node_2}' (sim={best_similarity_2})")

    # Use shortest path (with optional 2-hop extension)
    result = heuristic_path_with_embeddings(
        G, tokenizer, model, best_node_1, best_node_2,
        node_embeddings, top_k=3, second_hop=second_hop,
        data_dir=data_dir, save_files=save_files, verbatim=verbatim
    )
    if not result or result[0] is None:
        return None
    (path, path_graph, shortest_path_length, fname, graph_GraphML) = result
    return (best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML

def describe_communities(G, N=10):
    """Detect and describe the top N communities in graph G using Louvain method."""
    partition = community_louvain.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    sorted_communities = sorted(communities.items(), key=lambda item: len(item[1]), reverse=True)[:N]
    for i, (comm_id, nodes) in enumerate(sorted_communities, start=1):
        subgraph = G.subgraph(nodes)
        degrees = subgraph.degree()
        sorted_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)
        key_nodes = sorted_nodes[:5]
        print(f"Community {i} (ID {comm_id}) with {len(nodes)} nodes, key nodes:")
        for node_id, degree in key_nodes:
            print(f"  Node {node_id}: Degree {degree}")

def describe_communities_with_plots(G, N=10, N_nodes=5, data_dir='./'):
    """
    Detect and describe the top N communities with plots of community sizes and top-degree nodes.
    """
    partition = community_louvain.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    all_communities_sorted = sorted(communities.values(), key=len, reverse=True)
    all_sizes = [len(c) for c in all_communities_sorted]
    # Plot sizes of all communities
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(all_sizes)), all_sizes)
    plt.title('Size of All Communities')
    plt.xlabel('Community Index')
    plt.ylabel('Size (Number of Nodes)')
    plt.savefig(f'{data_dir}/size_of_communities.svg')
    plt.close()

    rows = math.ceil(N / 2)
    cols = 2 if N > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 10*rows), squeeze=False)
    for idx, nodes in enumerate(all_communities_sorted[:N]):
        subgraph = G.subgraph(nodes)
        degrees = dict(subgraph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        key_nodes, key_degrees = zip(*sorted_nodes[:N_nodes])
        ax = axes[idx // cols, idx % cols]
        ax.bar(range(len(key_nodes)), key_degrees, tick_label=key_nodes)
        ax.set_title(f'Community {idx+1} (Top Nodes by Degree)', fontsize=12)
        ax.set_xlabel('Node')
        ax.set_ylabel('Degree')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/top_nodes_by_degree.svg')
    plt.close()

def describe_communities_with_plots_complex(G, N=10, N_nodes=5, data_dir='./'):
    """
    Detect and describe the top N communities with additional plots for average degree, clustering coefficient, and betweenness centrality.
    """
    partition = community_louvain.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    all_communities_sorted = sorted(communities.values(), key=len, reverse=True)
    all_sizes = [len(c) for c in all_communities_sorted]
    avg_degrees = []
    avg_clusterings = []
    top_betweenness_values = []
    for nodes in all_communities_sorted:
        subgraph = G.subgraph(nodes)
        degrees = dict(subgraph.degree())
        avg_degrees.append(np.mean(list(degrees.values())))
        avg_clusterings.append(nx.average_clustering(subgraph))
        betweenness = nx.betweenness_centrality(subgraph)
        top_betweenness = sorted(betweenness.values(), reverse=True)[:N_nodes]
        top_betweenness_values.append(np.mean(top_betweenness) if top_betweenness else 0)

    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    axs = axs.flatten()
    # Size distribution
    axs[0].bar(range(len(all_sizes)), all_sizes)
    axs[0].set_title('Size of All Communities')
    axs[0].set_xlabel('Community Index')
    axs[0].set_ylabel('Size')
    # Average degree
    axs[1].bar(range(len(avg_degrees)), avg_degrees)
    axs[1].set_title('Average Node Degree per Community')
    axs[1].set_xlabel('Community Index')
    axs[1].set_ylabel('Average Degree')
    # Average clustering coefficient
    axs[2].bar(range(len(avg_clusterings)), avg_clusterings)
    axs[2].set_title('Average Clustering Coefficient per Community')
    axs[2].set_xlabel('Community Index')
    axs[2].set_ylabel('Clustering Coefficient')
    # Average betweenness centrality (top nodes)
    axs[3].bar(range(len(top_betweenness_values)), top_betweenness_values)
    axs[3].set_title('Avg Betweenness Centrality (Top Nodes) per Community')
    axs[3].set_xlabel('Community Index')
    axs[3].set_ylabel('Betweenness Centrality')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/community_stats_overview.svg')
    plt.close()

def is_scale_free_simple(G, plot_distribution=True, data_dir='./'):
    """
    Determine if network G is scale-free using the powerlaw package (simple version).
    """
    import powerlaw
    degrees = sorted([d for _, d in G.degree() if d > 0], reverse=True)
    fit = powerlaw.Fit(degrees, discrete=True)
    alpha = fit.power_law.alpha
    sigma = fit.power_law.sigma
    if plot_distribution:
        plt.figure(figsize=(10, 6))
        fit.plot_pdf(color='b', linewidth=2)
        fit.power_law.plot_pdf(color='r', linestyle='--', linewidth=2)
        plt.xlabel('Degree')
        plt.ylabel('Probability')
        plt.title('Degree Distribution with Power-law Fit')
        plt.savefig(f'{data_dir}/degree_distribution_powerlaw.svg')
        plt.close()
    R, p_val = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    is_scale_free = R > 0 and p_val < 0.05
    print(f"Power-law alpha: {alpha}, sigma: {sigma}")
    print(f"Log-likelihood ratio (R): {R}, p-value: {p_val}")
    return is_scale_free, fit

def is_scale_free(G, plot_distribution=True, data_dir='./', manual_xmin=None):
    """
    Determine if network G is scale-free using the powerlaw package with optional manual xmin.
    """
    import powerlaw
    degrees = sorted([d for _, d in G.degree() if d > 0], reverse=True)
    if manual_xmin:
        fit = powerlaw.Fit(degrees, discrete=True, xmin=manual_xmin)
    else:
        fit = powerlaw.Fit(degrees, discrete=True)
    if plot_distribution:
        plt.figure(figsize=(10, 6))
        fit.plot_pdf(color='b', linewidth=2)
        fit.power_law.plot_pdf(color='r', linestyle='--', linewidth=2)
        plt.xlabel('Degree')
        plt.ylabel('Probability')
        plt.title('Degree Distribution with Power-law Fit')
        plt.savefig(f'{data_dir}/degree_distribution_powerlaw_full.svg')
        plt.close()
    R, p_val = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    print(f"Power-law alpha: {fit.power_law.alpha}, sigma: {fit.power_law.sigma}")
    print(f"Log-likelihood ratio (R): {R}, p-value: {p_val}")
    is_scale_free = R > 0 and p_val < 0.05
    return is_scale_free, fit
