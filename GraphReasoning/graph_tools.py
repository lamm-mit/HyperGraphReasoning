# import sys
# sys.path.insert(0, "/orcd/pool/007/istewart/hypergraph/GraphReasoning_SG/GraphReasoning")
import heapq
import copy
import torch
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import community as community_louvain
import networkx as nx
import hypernetx as hnx
import pandas as pd
import os
import random
from pyvis.network import Network
from tqdm.notebook import tqdm
from GraphReasoning.utils import *
from hypernetx import Hypergraph
palette = "hls"


# Function to generate embeddings
def generate_node_embeddings(nodes, tokenizer, model, embeddings = {}, device='cuda:0'):
    # embeddings = {}

    if tokenizer:
        if type(nodes) == str: # one node
            inputs = tokenizer(nodes,
            padding=True,
            truncation=True,
            return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            try:
                embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            except:
                embeddings = outputs.hidden_states[-1].mean(dim=1).detach().to(torch.float).cpu().numpy()
        else:
            for node in tqdm(nodes):
                inputs = tokenizer(str(node),
                padding=True,
                truncation=True,
                return_tensors="pt"
                ).to(device)
                outputs = model(**inputs)
                try:
                    embeddings[node] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
                except:
                    embeddings[node] = outputs.hidden_states[-1].mean(dim=1).detach().to(torch.float).cpu().numpy()
   
    else: #when a tokenizer is not provided such as when you use sentence transformers assumes that model.encode(...) is available 
        if type(nodes) == str: # one node
            embeddings = model.encode(nodes) #tokenizes and embeds all in one step
        else:
            for node in tqdm(nodes):
                embeddings[node] = model.encode(node) 
                
     
    return embeddings


def generate_hypernode_embeddings(nodes, tokenizer, model, embeddings=None, device='cuda:0'):
    # initialize embeddings dict if not provided
    if embeddings is None:
        embeddings = {}

    # If they passed in a HyperNetX hypergraph, grab its node list
    if isinstance(nodes, hnx.Hypergraph):
        node_iter = list(nodes.nodes)
    else:
        node_iter = nodes

    # If it's a single-node string, handle that case:
    if isinstance(node_iter, str):
        inputs = tokenizer(
            node_iter,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)
        try:
            return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        except AttributeError:
            return outputs.hidden_states[-1].mean(dim=1).detach().cpu().numpy()

    # Otherwise it's an iterable of nodes
    for node in tqdm(node_iter):
        text = str(node)
        if tokenizer:
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            try:
                emb = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
            except AttributeError:
                emb = outputs.hidden_states[-1].mean(dim=1).detach().cpu().numpy()
        else:
            # e.g., sentence-transformers style API
            emb = model.encode(text)
        embeddings[node] = emb

    return embeddings

# def regenerate_node_embeddings(graph, nodes_to_recalculate, tokenizer, model):  deprecated -> use generate_node_embeddings
#     """
#     Regenerate embeddings for specific nodes.
#     """
#     new_embeddings = {}
#     for node in tqdm(nodes_to_recalculate):
#         inputs = tokenizer(node, return_tensors="pt")
#         outputs = model(**inputs)
#         new_embeddings[node] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
#     return new_embeddings


import pickle #pickle is a Python module that lets you save complex objects (like dictionaries, lists, models, etc.) to disk in a binary format, and load them back later.

def save_embeddings(embeddings, file_path): #file_ath is something like {data_dir}/node_embeddings.pkl written in your main script 
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings
 
def find_best_fitting_node(keyword, embeddings, tokenizer, model):
 
    keyword_embedding = generate_node_embeddings(keyword, tokenizer, model).flatten()
    # keyword_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()  # Flatten to ensure 1-D
    
    # Calculate cosine similarity and find the best match
    best_node = None
    best_similarity = float('-inf')  # Initialize with negative infinity
    for node, embedding in embeddings.items():
        # Ensure embedding is 1-D
        embedding = embedding.flatten()  # Flatten to ensure 1-D
        similarity = 1 - cosine(keyword_embedding, embedding)  # Cosine similarity
        if similarity > best_similarity:
            best_similarity = similarity
            best_node = node
            
    return best_node, best_similarity


def find_best_fitting_node_list(keyword, embeddings, tokenizer, model, N_samples=5, similarity_threshold=0.9):
    keyword_embedding = generate_node_embeddings(keyword, tokenizer, model).flatten()

    # Initialize a min-heap
    min_heap = []
    heapq.heapify(min_heap)
    
    for node, embedding in embeddings.items():
        # Ensure embedding is 1-D
        embedding = embedding.flatten()  # Flatten to ensure 1-D
        similarity = 1 - cosine(keyword_embedding, embedding)  # Cosine similarity

        # If the heap is smaller than N_samples, just add the current node and similarity
        if len(min_heap) < N_samples:
            heapq.heappush(min_heap, (similarity, node))
        else:
            # If the current similarity is greater than the smallest similarity in the heap
            if similarity > min_heap[0][0]:
                heapq.heappop(min_heap)  # Remove the smallest
                heapq.heappush(min_heap, (similarity, node))  # Add the current node and similarity
                
    # Convert the min-heap to a sorted list in descending order of similarity
    best_nodes = sorted(min_heap, key=lambda x: -x[0])
    
    for (similarity, node) in best_nodes[1:]: 
        if similarity < similarity_threshold:
            best_nodes.remove((similarity, node))

    # Return a list of tuples (node, similarity)
    return [(node, similarity) for similarity, node in best_nodes]


# Example usage
def visualize_embeddings_2d(embeddings , data_dir='./'):
    # Generate embeddings
    #embeddings = generate_node_embeddings(graph, tokenizer, model)
    
    # Extract the embedding vectors
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)
    for i, node_id in enumerate(node_ids):
        plt.text(vectors_2d[i, 0], vectors_2d[i, 1], str(node_id), fontsize=9)
    plt.title('Node Embeddings Visualization')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.savefig(f'{data_dir}/node_embeddings_2d.svg')  # Save the figure as SVG
    plt.show()


def visualize_embeddings_2d_notext(embeddings, n_clusters=3, data_dir='./'):
    # Extract the embedding vectors
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    labels = kmeans.labels_
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, alpha=0.5, cmap='viridis')
    plt.title('Node Embeddings Visualization')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.savefig(f'{data_dir}/node_embeddings_2d_clusters.svg')  # Save the figure as SVG
    plt.show()


def visualize_embeddings_2d_pretty(embeddings, n_clusters=3,  data_dir='./'):
    # Extract the embedding vectors
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    labels = kmeans.labels_
    
    # Count the number of points in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid')  # Set seaborn style for prettier plots
    
    # Use seaborn's color palette and matplotlib's scatter plot
    palette = sns.color_palette("hsv", n_clusters)  # Use a different color palette
    for cluster in range(n_clusters):
        cluster_points = vectors_2d[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster} (n={cluster_counts[cluster]})', alpha=0.7, edgecolors='w', s=100, cmap=palette)
    
    plt.title('Node Embeddings Visualization with Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(scatterpoints=1)  # Add a legend to show cluster labels and counts
    plt.savefig(f'{data_dir}/node_embeddings_2d_clusters_pretty.svg')  # Save the figure as SVG
    plt.show()
    
    # Optionally print the counts for each cluster
    for cluster, count in cluster_counts.items():
        print(f'Cluster {cluster}: {count} items')

from scipy.spatial.distance import cdist

def visualize_embeddings_2d_pretty_and_sample(embeddings, n_clusters=3, n_samples=5, data_dir='./',
                                             alpha=0.7, edgecolors='none', s=50,):
    # Extract the embedding vectors
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    labels = kmeans.labels_
    
    # Count the number of points in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid')  # Set seaborn style for prettier plots
    palette = sns.color_palette("hsv", n_clusters)
    for cluster in range(n_clusters):
        cluster_points = vectors_2d[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster} (n={cluster_counts[cluster]})'
                    , alpha=alpha, edgecolors=edgecolors, s=s, cmap=palette,#alpha=0.7, edgecolors='w', s=100, cmap=palette)
                   )
    
    plt.title('Node Embeddings Visualization with Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(scatterpoints=1)
    plt.savefig(f'{data_dir}/node_embeddings_2d_clusters_pretty.svg')
    plt.show()
    
    # Output N_sample terms from the center of each cluster
    centroids = kmeans.cluster_centers_
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_vectors = vectors[cluster_indices]
        cluster_node_ids = np.array(node_ids)[cluster_indices]
        
        # Calculate distances of points in this cluster to the centroid
        distances = cdist(cluster_vectors, [centroids[cluster]], 'euclidean').flatten()
        
        # Get indices of N_samples closest points
        closest_indices = np.argsort(distances)[:n_samples]
        closest_node_ids = cluster_node_ids[closest_indices]
        
        print(f'Cluster {cluster}: {len(cluster_vectors)} items')
        print(f'Closest {n_samples} node IDs to centroid:', closest_node_ids)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def visualize_embeddings_with_gmm_density_voronoi_and_print_top_samples(embeddings, n_clusters=5, top_n=3, data_dir='./',s=50):
    # Extract the embedding vectors
    descriptions = list(embeddings.keys())
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(vectors_2d)
    labels = gmm.predict(vectors_2d)
    
    # Generate Voronoi regions
    vor = Voronoi(gmm.means_)
    
    # Plotting
    plt.figure(figsize=(10, 10))
    # Plot Voronoi diagram
    voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors='black', line_width=1, line_alpha=0.7, point_size=2)
    
    # Color points based on their cluster
    for i in range(n_clusters):
        plt.scatter(vectors_2d[labels == i, 0], vectors_2d[labels == i, 1], s=s, label=f'Cluster {i}')
    
    plt.title('Embedding Vectors with GMM Density and Voronoi Tessellation')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.savefig(f'{data_dir}/node_embeddings_2d_clusters_voronoi.svg')
    
    plt.show()
    # Print top-ranked sample texts
    for i in range(n_clusters):
        cluster_center = gmm.means_[i]
        cluster_points = vectors_2d[labels == i]
        
        distances = euclidean_distances(cluster_points, [cluster_center])
        distances = distances.flatten()
        
        closest_indices = np.argsort(distances)[:top_n]
        
        print(f"\nTop {top_n} closest samples to the center of Cluster {i}:")
        for idx in closest_indices:
            original_idx = np.where(labels == i)[0][idx]
            desc = descriptions[original_idx]
            print(f"- Description: {desc}, Distance: {distances[idx]:.2f}")

def analyze_network(G,  data_dir='./', root = 'graph_analysis'):
    # Compute the degrees of the nodes
    # Compute the degrees of the nodes
    degrees = [d for n, d in G.degree()]
    
    # Compute maximum, minimum, and median node degrees
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = np.median(degrees)
    
    # Number of nodes and edges
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Average node degree
    avg_degree = np.mean(degrees)
    
    # Density of the network
    density = nx.density(G)
    
    # Number of communities (using connected components as a simple community proxy)
    num_communities = nx.number_connected_components(G)
    
    # Print the results
    print(f"Maximum Degree: {max_degree}")
    print(f"Minimum Degree: {min_degree}")
    print(f"Median Degree: {median_degree}")
    print(f"Number of Nodes: {num_nodes}")
    print(f"Number of Edges: {num_edges}")
    print(f"Average Node Degree: {avg_degree:.2f}")
    print(f"Density: {density:.4f}")
    print(f"Number of Communities: {num_communities}")
    
    # Plot the results
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))

    metrics = [
        ('Number of Nodes', num_nodes),
        ('Number of Edges', num_edges),
        ('Avg Node Degree', avg_degree),
        ('Density', density),
        ('Number of Communities', num_communities)
    ]
    
    for ax, (label, value) in zip(axs, metrics):
        ax.barh(label, value, color='blue')
        ax.set_xlim(0, max(value * 1.1, 1.1))  # Adding some padding for better visualization
        ax.set_xlabel('Value')
        ax.set_title(label)
    
    plt.tight_layout()
    plt.savefig(f'{data_dir}/community_structure_{root}.svg')
    # Show the plot
    plt.show()
    
    return max_degree, min_degree, median_degree

def graph_statistics_and_plots(G, data_dir='./'):
    # Calculate statistics
    degrees = [degree for node, degree in G.degree()]
    degree_distribution = np.bincount(degrees)
    average_degree = np.mean(degrees)
    clustering_coefficients = nx.clustering(G)
    average_clustering_coefficient = nx.average_clustering(G)
    triangles = sum(nx.triangles(G).values()) / 3
    connected_components = nx.number_connected_components(G)
    density = nx.density(G)
    
    # Diameter and Average Path Length (for connected graphs or components)
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        average_path_length = nx.average_shortest_path_length(G)
    else:
        diameter = "Graph not connected"
        component_lengths = [nx.average_shortest_path_length(G.subgraph(c)) for c in nx.connected_components(G)]
        average_path_length = np.mean(component_lengths)
    
    # Plot Degree Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), alpha=0.75, color='blue')
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.savefig(f'{data_dir}/degree_distribution.svg')
    #plt.close()
    plt.show()
    
    # Plot Clustering Coefficient Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(list(clustering_coefficients.values()), bins=10, alpha=0.75, color='green')
    plt.title('Clustering Coefficient Distribution')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')
    plt.savefig(f'{data_dir}/clustering_coefficient_distribution.svg')
    plt.show()
    #plt.close()
    
    statistics = {
        'Degree Distribution': degree_distribution,
        'Average Degree': average_degree,
        'Clustering Coefficients': clustering_coefficients,
        'Average Clustering Coefficient': average_clustering_coefficient,
        'Number of Triangles': triangles,
        'Connected Components': connected_components,
        'Diameter': diameter,
        'Density': density,
        'Average Path Length': average_path_length,
    }
    
    return statistics
 
def graph_statistics_and_plots_for_large_graphs (G, data_dir='./', include_centrality=False,
                                                 make_graph_plot=False,root='graph', log_scale=True, 
                                                 log_hist_scale=True,density_opt=False, bins=50,
                                                ):
    # Basic statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = [degree for node, degree in G.degree()]
    log_degrees = np.log1p(degrees)  # Using log1p for a better handle on zero degrees
    #degree_distribution = np.bincount(degrees)
    average_degree = np.mean(degrees)
    density = nx.density(G)
    connected_components = nx.number_connected_components(G)
    
    # Centrality measures
    if include_centrality:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Community detection with Louvain method
    partition = community_louvain.best_partition(G)
    num_communities = len(set(partition.values()))

    # Plotting
    # Degree Distribution on a log-log scale
    plt.figure(figsize=(10, 6))
     
    if log_scale:
        counts, bins, patches = plt.hist(log_degrees, bins=bins, alpha=0.75, color='blue', log=log_hist_scale, density=density_opt)
    
        plt.xscale('log')
        plt.yscale('log')
        xlab_0='Log(1 + Degree)'
        if density_opt:
            ylab_0='Probability Distribution'
        else: 
            ylab_0='Probability Distribution'
        ylab_0=ylab_0 + log_hist_scale*' (log)'    
        
        
        plt_title='Histogram of Log-Transformed Node Degrees with Log-Log Scale'
        
    else:
        counts, bins, patches = plt.hist(degrees, bins=bins, alpha=0.75, color='blue', log=log_hist_scale, density=density_opt)
        xlab_0='Degree'
        if density_opt:
            ylab_0='Probability Distribution'
        else: 
            ylab_0='Probability Distribution'
        ylab_0=ylab_0 + log_hist_scale*' (log)'     
        plt_title='Histogram of Node Degrees'

    plt.title(plt_title)
    plt.xlabel(xlab_0)
    plt.ylabel(ylab_0)
    plt.savefig(f'{data_dir}/{plt_title}_{root}.svg')
    plt.show()
    
    if make_graph_plot:
        
        # Additional Plots
        # Plot community structure
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G)  # for better visualization
        cmap = plt.get_cmap('viridis')
        nx.draw_networkx(G, pos, node_color=list(partition.values()), node_size=20, cmap=cmap, with_labels=False)
        plt.title('Community Structure')
        plt.savefig(f'{data_dir}/community_structure_{root}.svg')
        plt.show()
        plt.close()

    # Save statistics
    statistics = {
        'Number of Nodes': num_nodes,
        'Number of Edges': num_edges,
        'Average Degree': average_degree,
        'Density': density,
        'Connected Components': connected_components,
        'Number of Communities': num_communities,
        # Centrality measures could be added here as well, but they are often better analyzed separately due to their detailed nature
    }
    if include_centrality:
        centrality = {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality,
            'eigenvector_centrality': eigenvector_centrality,
        }
    else:
        centrality=None
 
    
    return statistics, include_centrality

## Now add these colors to communities and make another dataframe
def colors2Community(communities) -> pd.DataFrame:
    ## Define a color palette
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors


 
def graph_Louvain (G, 
                  graph_GraphML=None, palette = "hls"):
    # Assuming G is your graph and data_dir is defined
    
    # Compute the best partition using the Louvain algorithm
    G_undir = G.to_undirected()
    partition = community_louvain.best_partition(G_undir)

    # Organize nodes into communities based on the Louvain partition
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    
    communities_list = list(communities.values())
    print("Number of Communities =", len(communities_list))
    print("Communities: ", communities_list)
    
    # Assuming colors2Community can work with the communities_list format
    colors = colors2Community(communities_list)
    print("Colors: ", colors)
    
    # Assign attributes to nodes based on their community membership
    for index, row in colors.iterrows():
        node = row['node']
        G.nodes[node]['group'] = row['group']
        G.nodes[node]['color'] = row['color']
        G.nodes[node]['size'] = G.degree[node]
    
    print("Done, assigned colors and groups...")
    
    # Write the graph with community information to a GraphML file
    if graph_GraphML != None:
        try:
            nx.write_graphml(G, graph_GraphML)
    
            print("Written GraphML.")

        except:
            print ("Error saving GraphML file.")
    return G
    
def save_graph (G, 
                  graph_GraphML=None, ):
    if graph_GraphML != None:
        nx.write_graphml(G, graph_GraphML)
    
        print("Written GraphML")
    else:
        print("Error, no file name provided.")
    return 

def update_node_embeddings(embeddings, graph_new, tokenizer, model, remove_embeddings_for_nodes_no_longer_in_graph=True,
                          verbatim=False):
    """
    Update embeddings for new nodes in an updated graph, ensuring that the original embeddings are not altered.

    Args:
    - embeddings (dict): Existing node embeddings.
    - graph_new: The updated graph object.
    - tokenizer: Tokenizer object to tokenize node names.
    - model: Model object to generate embeddings.

    Returns:
    - Updated embeddings dictionary with embeddings for new nodes, without altering the original embeddings.
    """
    # Create a deep copy of the original embeddings
    embeddings_updated = copy.deepcopy(embeddings)
    
    # embeddings_new = generate_node_embeddings(graph_new.nodes(), tokenizer, model)
    # # Iterate through new graph nodes
    # for node in tqdm(graph_new.nodes()):
    #     # Check if the node already has an embedding in the copied dictionary
    #     if node not in embeddings_updated:
    #         if verbatim:
    #             print(f"Generating embedding for new node: {node}")
    #         inputs = tokenizer(node, return_tensors="pt")
    #         outputs = model(**inputs)
    #         # Update the copied embeddings dictionary with the new node's embedding
    #         embeddings_updated[node] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        # Remove embeddings for nodes that no longer exist in the graph from the copied dictionary
    nodes_in_graph = set(graph_new.nodes())
    for node in nodes_in_graph:
        if node not in list(embeddings_updated):
            if verbatim:
                print(f"Adding embedding for node: {node}")
            embeddings_updated[node] = generate_node_embeddings(node, tokenizer, model)
            
    if remove_embeddings_for_nodes_no_longer_in_graph:
        # Remove embeddings for nodes that no longer exist in the graph from the copied dictionary
        for node in list(embeddings_updated):
            if node not in nodes_in_graph:
                if verbatim:
                    print(f"Removing embedding for node no longer in graph: {node}")
                del embeddings_updated[node]

    return embeddings_updated


def update_hypernode_embeddings(
    embeddings,
    graph_new,
    tokenizer,
    model,
    remove_embeddings_for_nodes_no_longer_in_graph=True,
    verbatim=False
):
    """
    Update embeddings for new nodes in an updated graph (NetworkX or HyperNetX),
    without mutating the original `embeddings` dict.

    Args:
      embeddings (dict): Existing node→embedding mapping.
      graph_new:   A NetworkX Graph/DiGraph, a HyperNetX Hypergraph, or any iterable of node IDs.
      tokenizer:   Tokenizer for your model (or None).
      model:       Your embedding model.
      remove_embeddings_for_nodes_no_longer_in_graph (bool): prune old nodes.
      verbatim (bool): print progress messages.

    Returns:
      dict: A fresh dict with embeddings for all nodes in `graph_new`.
    """
    embeddings_updated = copy.deepcopy(embeddings or {})

    # 2) Extract the set of nodes, handling several types
    if isinstance(graph_new, hnx.Hypergraph):
        nodes_in_graph = set(graph_new.nodes)
    elif hasattr(graph_new, "nodes"):
        # e.g. networkx Graph or DiGraph
        nodes_in_graph = set(graph_new.nodes())
    else:
        # assume it's already an iterable of node IDs
        nodes_in_graph = set(graph_new)

    # 3) Find which nodes need embeddings
    new_nodes = [n for n in nodes_in_graph if n not in embeddings_updated]
    if new_nodes:
        if verbatim:
            print(f"Generating embeddings for {len(new_nodes)} new node(s)")
        # Batch‐generate all at once
        new_embs = generate_hypernode_embeddings(new_nodes, tokenizer, model)
        embeddings_updated.update(new_embs)

    # 4) Optionally prune embeddings for nodes no longer in the graph
    if remove_embeddings_for_nodes_no_longer_in_graph:
        for n in list(embeddings_updated):
            if n not in nodes_in_graph:
                if verbatim:
                    print(f"Pruning embedding for removed node: {n}")
                del embeddings_updated[n]

    return embeddings_updated



def remove_small_fragents (G_new, size_threshold):
    if size_threshold >0:
        
        # Find all connected components, returned as sets of nodes
        try:
            components = list(nx.connected_components(G_new))
        except:
            print("using weakly connected components...")
            components = list(nx.weakly_connected_components(G_new))
        # Iterate through components and remove those smaller than the threshold
        for component in components:
            if len(component) < size_threshold:
                # Remove the nodes in small components
                G_new.remove_nodes_from(component)
    return G_new

# def remove_small_hyperfragments(
#     H_new: hnx.Hypergraph,
#     size_threshold: int,
#     return_singletons: bool = False,
#     sub_dfs: sub_dfs, 
# ) -> hnx.Hypergraph:
#     """
#     Remove all nodes belonging to node-connected components smaller than size_threshold.

#     Args:
#       H_new:           Original HyperNetX hypergraph.
#       size_threshold:  Minimum component size (in nodes) to keep.
#       return_singletons: If True, keep singleton components (size=1).

#     Returns:
#       A new HyperNetX Hypergraph with small fragments removed.
#     """
#     if size_threshold <= 0:
#         return H_new

#     # 1) Gather all node components (each is a set of node IDs)
#     comps = list(H_new.connected_components())  # same as s_connected_components(s=1, edges=False)

#     # 2) Identify which nodes lie in undersized components
#     to_remove = {
#         node
#         for comp in comps
#         if len(comp) < size_threshold and not (return_singletons and len(comp) == 1)
#         for node in comp
#     }
#     if not to_remove:
#         return H_new

#     # 3) Build the set of nodes to keep
#     keep_nodes = set(H_new.nodes) - to_remove

#     # 4) Restrict the hypergraph to those nodes
#     H_reduced = H_new.restrict_to_nodes(keep_nodes)

#     return H_reduced

def remove_small_hyperfragments(
    H_new: hnx.Hypergraph,
    sub_dfs: list[pd.DataFrame],
    size_threshold: int,
    return_singletons: bool = False,
) -> tuple[hnx.Hypergraph, list[pd.DataFrame]]:
    """
    Remove all nodes belonging to node‑connected components smaller than size_threshold,
    and prune the corresponding rows from each DataFrame in sub_dfs.

    Args:
      H_new:             Original HyperNetX hypergraph.
      sub_dfs:           List of DataFrames, each containing a 'chunk_id' column.
      size_threshold:    Minimum component size (in nodes) to keep.
      return_singletons: If True, keep singleton components (size=1).

    Returns:
      H_reduced:         Hypergraph with small fragments removed.
      pruned_sub_dfs:    List of DataFrames with rows for pruned nodes dropped.
    """
    # trivial cases: nothing to prune
    if size_threshold <= 0:
        return H_new, sub_dfs

    # 1) find all connected components
    comps = list(H_new.connected_components())

    # 2) collect nodes in undersized components
    to_remove = {
        node
        for comp in comps
        if len(comp) < size_threshold and not (return_singletons and len(comp) == 1)
        for node in comp
    }
    if not to_remove:
        return H_new, sub_dfs

    # 3) restrict hypergraph
    keep_nodes = set(H_new.nodes) - to_remove
    H_reduced = H_new.restrict_to_nodes(keep_nodes)

    # 4) prune each DataFrame by chunk_id
    updated_sub_dfs = []
    for df in sub_dfs:
        # prune out 
        pruned = df[
            (~df['source'].isin(to_remove)) &
            (~df['target'].isin(to_remove))
        ].reset_index(drop=True)
        updated_sub_dfs.append(pruned)

    return H_reduced, updated_sub_dfs


def simplify_node_name_with_llm(node_name, generate, max_tokens=2048, temperature=0.3):
    # Generate a prompt for the LLM to simplify or describe the node name
    system_prompt='You are an ontological graph maker. You carefully rename nodes in complex networks.'
    prompt = f"Provide a simplified, more descriptive name for a network node named '{node_name}' that reflects its importance or role within a network."
   
    # Assuming 'generate' is a function that calls the LLM with the given prompt
    #simplified_name = generate(system_prompt=system_prompt, prompt)
    simplified_name = generate(system_prompt=system_prompt, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
   
    return simplified_name

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
from powerlaw import Fit

# Assuming regenerate_node_embeddings is defined as provided earlier

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# def simplify_node_name_with_llm(node_name, max_tokens, temperature):
#     # This is a placeholder for the actual function that uses a language model
#     # to generate a simplified or more descriptive node name.
#     return node_name  

def simplify_graph(graph_, node_embeddings, tokenizer, model, similarity_threshold=0.9, use_llm=False,
                   data_dir_output='./', graph_root='simple_graph', verbatim=False, max_tokens=2048, 
                   temperature=0.3, generate=None):
    """
    Simplifies a graph by merging similar nodes and optionally renaming them using a language model.
    """

    graph = graph_.copy()
    
    nodes = list(node_embeddings.keys())
    nodes_png = []
    
    for node in nodes:
        if '.png' in node:
            nodes_png.append(node)
            
    for node_ in nodes_png:
        nodes.remove(node_)

    for node in nodes:
        if '.png' in node:
            print(node)

    embeddings_matrix = np.array([node_embeddings[node].flatten() for node in nodes])

    similarity_matrix = cosine_similarity(embeddings_matrix)
    to_merge = np.where(similarity_matrix > similarity_threshold)

    node_mapping = {}
    nodes_to_recalculate = set()
    merged_nodes = set()  # Keep track of nodes that have been merged
    if verbatim:
        print("Start...")
    for i, j in tqdm(zip(*to_merge), total=len(to_merge[0])):
        if i != j and nodes[i] not in merged_nodes and nodes[j] not in merged_nodes:  # Check for duplicates
            node_i, node_j = nodes[i], nodes[j]
            
            try:
                if graph.degree(node_i) >= graph.degree(node_j):
                #if graph.degree[node_i] >= graph.degree[node_j]:
                    node_to_keep, node_to_merge = node_i, node_j
                else:
                    node_to_keep, node_to_merge = node_j, node_i
    
                if verbatim:
                    print("Node to keep and merge:", node_to_keep, "<--", node_to_merge)
    
                #if use_llm and node_to_keep in nodes_to_recalculate:
                #    node_to_keep = simplify_node_name_with_llm(node_to_keep, max_tokens=max_tokens, temperature=temperature)
    
                node_mapping[node_to_merge] = node_to_keep
                nodes_to_recalculate.add(node_to_keep)
                merged_nodes.add(node_to_merge)  # Mark the merged node to avoid duplicate handling
            except:
                print (end="")
    if verbatim:
        print ("Now relabel. ")
    # Create the simplified graph by relabeling nodes.
    new_graph = nx.relabel_nodes(graph, node_mapping, copy=True)
    if verbatim:
        print ("New graph generated, nodes relabled. ")
    new_graph.subgraph(nodes_to_recalculate)
    # Recalculate embeddings for nodes that have been merged or renamed.
    recalculated_embeddings = generate_node_embeddings(new_graph.nodes(), tokenizer, model)
    if verbatim:
        print ("Relcaulated embeddings... ")
    # Update the embeddings dictionary with the recalculated embeddings.
    updated_embeddings = {**node_embeddings, **recalculated_embeddings}

    # Remove embeddings for nodes that no longer exist in the graph.
    for node in merged_nodes:
        updated_embeddings.pop(node, None)
    if verbatim:
        print ("Now save graph... ")

    # Save the simplified graph to a file.
    graph_path = f'{data_dir_output}/{graph_root}_graphML_simplified.graphml'
    nx.write_graphml(new_graph, graph_path)

    if verbatim:
        print(f"Graph simplified and saved to {graph_path}")

    return new_graph, updated_embeddings


def simplify_hypergraph(
    graph_,
    sub_dfs,
    node_embeddings, 
    tokenizer, 
    model, 
    similarity_threshold=0.9,
    use_llm=False,
    data_dir_output='./',
    graph_root='simple_hypergraph', 
    verbatim=False, 
    device='cuda:0'
):
    """
    Simplify a HyperNetX hypergraph by merging similar nodes.
    
    Returns:
      new_graph (hnx.Hypergraph), updated_embeddings (dict)
    """
    updated_sub_dfs = []
    # 1) Deep-copy embeddings & original hypergraph
    embeddings_updated = copy.deepcopy(node_embeddings)
    H = copy.deepcopy(graph_)
    df_list = [df.copy() for df in sub_dfs]  # work on copies

    # 2) Filter out any image-nodes
    nodes = [n for n in H.nodes if not str(n).endswith('.png')]
    if verbatim:
        print(f"Computing similarity among {len(nodes)} nodes…")

    # 3) Build similarity matrix
    emb_mat = np.vstack([embeddings_updated[n].flatten() for n in nodes])
    sim = cosine_similarity(emb_mat)

    # 4) Identify merge pairs (upper triangle only)
    to_merge = np.where(np.triu(sim, k=1) > similarity_threshold)
    node_mapping = {}
    merged = set()
    keepers = set()

    for i, j in zip(*to_merge):
        ni, nj = nodes[i], nodes[j]
        if ni in merged or nj in merged:
            continue
        # choose the node with higher degree to keep
        di, dj = H.degree(ni), H.degree(nj)
        keep, remove = (ni, nj) if di >= dj else (nj, ni)
        node_mapping[remove] = keep
        merged.add(remove)
        keepers.add(keep)
        if verbatim:
            print(f"Merging '{remove}' → '{keep}'")

    # 5) Rebuild incidence dict with merged nodes collapsed
    new_incidence = {}
    for hedge, members in H.incidence_dict.items():
        new_members = { node_mapping.get(n, n) for n in members }
        if len(new_members) > 1:
            new_incidence[hedge] = list(new_members)

    new_graph = hnx.Hypergraph(new_incidence)
    if verbatim:
        print(f"Simplified hypergraph: {len(new_graph.nodes)} nodes, "
              f"{len(new_graph.edges)} hyperedges")

    # 6) Recompute embeddings for any keepers that gained new members
    if keepers:
        if verbatim:
            print(f"Recomputing embeddings for {len(keepers)} merged-into nodes")
        #new_embs = generate_node_embeddings( COMMENTING OUT FOR NOW 
        new_embs = generate_hypernode_embeddings(
            list(keepers), tokenizer, model, device=device
        )
        embeddings_updated.update(new_embs)

    # 7) Prune embeddings of removed nodes
    for rem in merged:
        embeddings_updated.pop(rem, None)
        if verbatim:
            print(f"Pruned embedding for '{rem}'")
    
    #Fix subdfs
    try:
        updated_sub_dfs = []
        for df in df_list:
            df[['source','target']] = df[['source','target']].replace(node_mapping)
            df = df[df.source != df.target].reset_index(drop=True)
            updated_sub_dfs.append(df)
    except Exception as e:
        print(f"[simplify] couldn’t remap sub_dfs—falling back: {e!r}")
        updated_sub_dfs = df_list

    # 8) Save simplified hypergraph
    path = f"{data_dir_output}/{graph_root}_simplified.pkl"
    with open(path, 'wb') as f:
        pickle.dump(new_graph, f)
    if verbatim:
        print(f"Simplified hypergraph saved to {path}")

    return new_graph, embeddings_updated, updated_sub_dfs



def make_HTML (G,data_dir='./', graph_root='graph_root'):

    net = Network(
            #notebook=False,
            notebook=True,
            # bgcolor="#1a1a1a",
            cdn_resources="remote",
            height="900px",
            width="100%",
            select_menu=True,
            # font_color="#cccccc",
            filter_menu=False,
        )
        
    net.from_nx(G)
    # net.repulsion(node_distance=150, spring_length=400)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
    # net.barnes_hut(gravity=-18100, central_gravity=5.05, spring_length=380)
    
    #net.show_buttons(filter_=["physics"])
    net.show_buttons()
    
    #net.show(graph_output_directory, notebook=False)
    graph_HTML= f'{data_dir}/{graph_root}_graphHTML.html'
    
    net.show(graph_HTML, #notebook=True
            )

    return graph_HTML

def return_giant_component_of_graph (G_new ):
    connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
    G_new = G_new.subgraph(connected_components[0]).copy()
    return G_new 
    
def return_giant_component_G_and_embeddings (G_new, node_embeddings):
    connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
    G_new = G_new.subgraph(connected_components[0]).copy()
    node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)
    return G_new, node_embeddings

def extract_number(filename):
    # This function extracts numbers from a filename and converts them to an integer.
    # It finds all sequences of digits in the filename and returns the first one as an integer.
    # If no number is found, it returns -1.
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else -1
 
def get_list_of_graphs_and_chunks (graph_q='graph_*_graph_clean.csv',  chunk_q='graph_*_chunks_clean.csv', data_dir='./',verbatim=False):
    graph_pattern = os.path.join(data_dir, graph_q)
    chunk_pattern = os.path.join(data_dir, chunk_q)
    
    # Use glob to find all files matching the patterns
    graph_files = glob.glob(graph_pattern)
    chunk_files = glob.glob(chunk_pattern)
    
    # Sort the files using the custom key function
    graph_file_list = sorted(graph_files, key=extract_number)
    chunk_file_list = sorted(chunk_files, key=extract_number)

    if verbatim:
        # Print the lists to verify
        print ('\n'.join(graph_file_list[:10]), '\n\n', '\n'.join(chunk_file_list[:10]),'\n')
        
        print('# graph files:', len (graph_file_list))
        print('# chunk files:', len (chunk_file_list))
    
    return graph_file_list, chunk_file_list

def print_graph_nodes_with_texts(G, separator="; ", N=64):
    """
    Prints out each node in the graph along with the associated texts, concatenated into a single string.

    Parameters:
    - G: A NetworkX graph object where each node has a 'texts' attribute containing a list of texts.
    - separator: A string separator used to join texts. Default is "; ".
    """
    print("Graph Nodes and Their Associated Texts (Concatenated):")
    for node, data in G.nodes(data=True):
        texts = data.get('texts', [])
        concatenated_texts = separator.join(texts)
        print(f"Node: {node}, Texts: {concatenated_texts[:N]}")      
       
def print_graph_nodes (G, separator="; ", N=64):
    """
    Prints out each node in the graph along with the associated texts, concatenated into a single string.

    Parameters:
    - G: A NetworkX graph object where each node has a 'texts' attribute containing a list of texts.
    - separator: A string separator used to join texts. Default is "; ".
    """
    i=0
    print("Graph Nodes and Their Associated Texts (Concatenated):")
    for node in G.nodes :
        print(f"Node {i}: {node}")  
        i=i+1
def get_text_associated_with_node(G, node_identifier ='bone', ):
        
    # Accessing and printing the 'texts' attribute for the node
    if 'texts' in G.nodes[node_identifier]:
        texts = G.nodes[node_identifier]['texts']
        concatenated_texts = "; ".join(texts)  # Assuming you want to concatenate the texts
        print(f"Texts associated with node '{node_identifier}': {concatenated_texts}")
    else:
        print(f"No 'texts' attribute found for node {node_identifier}")
        concatenated_texts=''
    return concatenated_texts 

import networkx as nx
import json
from copy import deepcopy
from tqdm import tqdm

def save_graph_with_text_as_JSON(G_or, data_dir='./', graph_name='my_graph.graphml'):
    G = deepcopy(G_or)

    # Ensure correct path joining
    import os
    fname = os.path.join(data_dir, graph_name)

    for _, data in tqdm(G.nodes(data=True)):
        for key in data:
            if isinstance(data[key], (list, dict, set, tuple)):  # Extend this as needed
                data[key] = json.dumps(data[key])

    for _, _, data in tqdm(G.edges(data=True)):
        for key in data:
            if isinstance(data[key], (list, dict, set, tuple)):  # Extend this as needed
                data[key] = json.dumps(data[key])

    nx.write_graphml(G, fname)
    return fname

def load_graph_with_text_as_JSON(data_dir='./', graph_name='my_graph.graphml'):
    # Ensure correct path joining
    import os
    fname = os.path.join(data_dir, graph_name)

    G = nx.read_graphml(fname)

    for node, data in tqdm(G.nodes(data=True)):
        for key, value in data.items():
            if isinstance(value, str):  # Only attempt to deserialize strings
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass  # If the value is not a valid JSON string, do nothing

    for _, _, data in tqdm(G.edges(data=True)):
        for key, value in data.items():
            if isinstance(value, str):
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass

    return G

from copy import deepcopy
import networkx as nx
from tqdm import tqdm
import os

def save_graph_without_text(G_or, data_dir='./', graph_name='my_graph.graphml'):
    G = deepcopy(G_or)

    # Process nodes: remove 'texts' attribute and convert others to string
    for _, data in tqdm(G.nodes(data=True), desc="Processing nodes"):
        if 'texts' in data:
            del data['texts']  # Remove the 'texts' attribute
        # Convert all other attributes to strings
        for key in data:
            data[key] = str(data[key])

    # Process edges: similar approach, remove 'texts' and convert attributes
    for i, (_, _, data) in enumerate(tqdm(G.edges(data=True), desc="Processing edges")):
    #for _, _, data in tqdm(G.edges(data=True), desc="Processing edges"):
        data['id'] = str(i)  # Assign a unique ID
        if 'texts' in data:
            del data['texts']  # Remove the 'texts' attribute
        # Convert all other attributes to strings
        for key in data:
            data[key] = str(data[key])
    
    # Ensure correct directory path and file name handling
    fname = os.path.join(data_dir, graph_name)
    
    # Save the graph to a GraphML file
    nx.write_graphml(G, fname, edge_id_from_attribute='id')
    return fname

def print_nodes_and_labels (G, N=10):
    # Printing out the first 10 nodes
    ch_list=[]
    
    print("First 10 nodes:")
    for node in list(G.nodes())[:10]:
        print(node)
    
    print("\nFirst 10 edges with titles:")
    for (node1, node2, data) in list(G.edges(data=True))[:10]:
        edge_title = data.get('title')  # Replace 'title' with the attribute key you're interested in
        ch=f"Node labels: ({node1}, {node2}) - Title: {edge_title}"
        ch_list.append (ch)
        
        print (ch)
        

    return ch_list

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import json
from tqdm import tqdm
import pandas as pd
import networkx as nx
import os 
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import json

def make_graph_from_text_withtext(graph_file_list, chunk_file_list,
                                  include_contextual_proximity=False,
                                  graph_root='graph_root',
                                  repeat_refine=0, verbatim=False,
                                  data_dir='./data_output_KG/',
                                  save_PDF=False, save_HTML=True, N_max=10,
                                  idx_start=0):
    """
    Constructs a graph from text data, ensuring edge labels do not incorrectly include node names.
    """

    # Initialize an empty DataFrame to store all texts
    all_texts_df = pd.DataFrame()

    # Initialize an empty graph
    G_total = nx.Graph()

    for idx in tqdm(range(idx_start, min(len(graph_file_list), N_max)), desc="Processing graphs"):
        try:
            # Load graph and chunk data
            graph_df = pd.read_csv(graph_file_list[idx])
            text_df = pd.read_csv(chunk_file_list[idx])
            
            # Append the current text_df to the all_texts_df
            all_texts_df = pd.concat([all_texts_df, text_df], ignore_index=True)
    
            # Clean and aggregate the graph data
            graph_df.replace("", np.nan, inplace=True)
            graph_df.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
            graph_df['count'] = 4  # Example fixed count, adjust as necessary
            
            # Aggregate edges and combine attributes
            graph_df = (graph_df.groupby(["node_1", "node_2"])
                        .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
                        .reset_index())
            
            if verbatim:
                print("Shape of graph DataFrame: ", graph_df.shape)
    
            # Add edges to the graph
            for _, row in graph_df.iterrows():
                G_total.add_edge(row['node_1'], row['node_2'], chunk_id=row['chunk_id'],
                                 title=row['edge'], weight=row['count'] / 4)
    
        except Exception as e:
            print(f"Error in graph generation for idx={idx}: {e}")
   
    # Ensure no duplicate chunk_id entries
    all_texts_df = all_texts_df.drop_duplicates(subset=['chunk_id'])
    
    # Map chunk_id to text
    chunk_id_to_text = pd.Series(all_texts_df.text.values, index=all_texts_df.chunk_id).to_dict()

    # Initialize node texts collection
    node_texts = {node: set() for node in G_total.nodes()}

    # Associate texts with nodes based on edges
    for (node1, node2, data) in tqdm(G_total.edges(data=True), desc="Mapping texts to nodes"):
        chunk_ids = data.get('chunk_id', '').split(',')
        for chunk_id in chunk_ids:
            text = chunk_id_to_text.get(chunk_id, "")
            if text:  # If text is found for the chunk_id
                node_texts[node1].add(text)
                node_texts[node2].add(text)

    # Update nodes with their texts
    for node, texts in node_texts.items():
        G_total.nodes[node]['texts'] = list(texts)  # Convert from set to list

    return G_total
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


    
def simplify_graph_with_text(graph_, node_embeddings, tokenizer, model, similarity_threshold=0.9, use_llm=False,
                   data_dir_output='./', graph_root='simple_graph', verbatim=False, max_tokens=2048, 
                   temperature=0.3, generate=None):
    """
    Simplifies a graph by merging similar nodes and optionally renaming them using a language model.
    Also, merges 'texts' node attribute ensuring no duplicates.
    """

    graph = deepcopy(graph_)
    
    nodes = list(node_embeddings.keys())
    for node in nodes:
        if '.png' in node:
            nodes.remove(node)
    embeddings_matrix = np.array([node_embeddings[node].flatten() for node in nodes])

    similarity_matrix = cosine_similarity(embeddings_matrix)
    to_merge = np.where(similarity_matrix > similarity_threshold)

    node_mapping = {}
    kept_nodes = set()
    merged_nodes = set()  # Keep track of nodes that have been merged
    if verbatim:
        print("Start...")
    for i, j in tqdm(zip(*to_merge), total=len(to_merge[0])):
        if i != j and nodes[i] not in merged_nodes and nodes[j] not in merged_nodes:  # Check for duplicates
            node_i, node_j = nodes[i], nodes[j]
            
            try:
                if graph.degree(node_i) >= graph.degree(node_j):
                    node_to_keep, node_to_merge = node_i, node_j
                else:
                    node_to_keep, node_to_merge = node_j, node_i
    
                # Handle 'texts' attribute by merging and removing duplicates
                texts_to_keep = set(graph.nodes[node_to_keep].get('texts', []))
                texts_to_merge = set(graph.nodes[node_to_merge].get('texts', []))
                merged_texts = list(texts_to_keep.union(texts_to_merge))
                graph.nodes[node_to_keep]['texts'] = merged_texts
    
                if verbatim:
                    print("Node to keep and merge:", node_to_keep, "<--", node_to_merge)
    
                node_mapping[node_to_merge] = node_to_keep
                kept_nodes.add(node_to_keep)
                merged_nodes.add(node_to_merge)  # Mark the merged node to avoid duplicate handling
            except Exception as e:
                print("Error during merging:", e)
    if verbatim:
        print ("Now relabel. ")
    # Create the simplified graph by relabeling nodes.
    new_graph = nx.relabel_nodes(graph, node_mapping, copy=True)
    if verbatim:
        print ("New graph generated, nodes relabled. ")
    # Recalculate embeddings for nodes that have been merged or renamed.
    new_graph=graph.subgraph(kept_nodes)
    new_embeddings = {k: node_embeddings[k] for k in new_graph.nodes()}
    # recalculated_embeddings = generate_node_embeddings(new_graph.nodes(), tokenizer, model)
    # if verbatim:
    #     print ("Relcaulated new embeddings... ")
    # Update the embeddings dictionary with the recalculated embeddings.
    updated_embeddings = {**node_embeddings, **new_embeddings}
    if verbatim:
        print ("Done renew embeddings... ")
    
    # Remove embeddings for nodes that no longer exist in the graph.
    for node in merged_nodes:
        if verbatim:
            print('Remove embeddings for nodes that no longer exist in the graph.')
        updated_embeddings.pop(node, None)
        
    if verbatim:
        print ("Now save graph... ")

    # Save the simplified graph to a file.
    graph_path = f'{graph_root}_graphML_simplified_JSON.graphml'
    save_graph_with_text_as_JSON (new_graph, data_dir=data_dir_output, graph_name=graph_path)
    
    if verbatim:
        print(f"Graph simplified and saved to {graph_path}")

    return new_graph, updated_embeddings


def find_shortest_path_subgraph_between_nodes(graph, nodes):
    subgraph=set()
    found=0
    all_path=0
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            all_path+=1
            try:
                path = nx.shortest_path(graph, nodes[i], nodes[j])
                print(f'Path between {nodes[i]}, {nodes[j]} found as {path}') 
                subgraph.update(path)
                found+=1
            except:
                print(f'No path between {nodes[i]}, {nodes[j]} found')
    print(f'Path found ratio = {found/all_path}')
    return graph.subgraph(list(subgraph))

from itertools import combinations
from collections import defaultdict, deque
import warnings

def find_shortest_path_hypersubgraph_between_nodes_local(H, nodes, s=1, k_paths=1):
    """
    Efficient version:
    For every unordered pair (u, v) in `nodes`, run a *local BFS over edges*
    starting from edges touching u until reaching edges touching v.
    
    This avoids building the global s-linegraph (O(E^2) cost) and instead performs 
    localized graph search using the inverted index node → incident edges.
    
    Returns:
        H_sub:  sub-hypergraph containing all edges in all discovered paths
        reports: detailed hop-level information compatible with collect_hyperentities
    """
    nodes = [str(n) for n in nodes if n is not None]

    # ---------------------------------------------------------
    # Build inverted index: node -> edges
    # ---------------------------------------------------------
    node_to_edges = defaultdict(set)
    for e in H.edges:
        for n in H.edges[e]:
            node_to_edges[str(n)].add(e)

    # ---------------------------------------------------------
    # Helper: check if two edges intersect in ≥ s nodes
    # ---------------------------------------------------------
    # def intersects(Ei, Ej):
    #     return len(H.edges[Ei].intersection(H.edges[Ej])) >= s

    def intersects(Ei, Ej):
        # --- Safe access for HyperNetX hyperedges ---
        try:
            A = H.edges[Ei]     # Works for HypergraphView
            B = H.edges[Ej]
        except KeyError:
            return False        # edge does not exist
        except Exception:
            return False        # some other weird failure
        
        # --- Convert to Python sets (HyperNetX returns Edge objects) ---
        try:
            A = set(A)
        except Exception:
            try:
                A = set(list(A))
            except Exception:
                return False
    
        try:
            B = set(B)
        except Exception:
            try:
                B = set(list(B))
            except Exception:
                return False
    
        # --- Overlap ≥ s check ---
        return len(A & B) >= s

    # ---------------------------------------------------------
    # Local BFS between two sets of edges
    # (edges touching u) → (edges touching v)
    # ---------------------------------------------------------
    def shortest_paths_between_edge_sets(S, T, k=k_paths):
        """
        Returns up to k shortest hyperedge paths (top-k), sorted by hop count.
        Supports multiple parents → multiple distinct shortest paths.
        """
    
        S = list(S)
        T = set(T)
        if not S or not T:
            return []

        q = deque()
        parents = defaultdict(list)
        depth = {}
        paths_found = []
        
        # Initialize BFS queue
        for se in S:
            q.append(se)
            depth[se] = 0
        
        min_depth_found = None
        
        while q and len(paths_found) < k:
            e = q.popleft()
            d = depth[e]
        
            # If we reached a target hyperedge (candidate path)
            if e in T:
                if min_depth_found is None:
                    min_depth_found = d
        
                if d == min_depth_found:
                    # reconstruct all possible shortest paths ending at e
                    def build_paths(node):
                        if not parents[node]:
                            return [[node]]
                        all_paths = []
                        for p in parents[node]:
                            for tail in build_paths(p):
                                all_paths.append(tail + [node])
                        return all_paths
        
                    new_paths = build_paths(e)
                    for p in new_paths:
                        paths_found.append(p)
                        if len(paths_found) >= k:
                            return sorted(paths_found, key=len)
        
                elif d > min_depth_found:
                    break
        
            # Expand neighbors
            for node in H.edges[e]:
                for nbr in node_to_edges[str(node)]:
                    if not intersects(e, nbr):
                        continue
        
                    nd = d + 1
        
                    # First discovery of nbr
                    if nbr not in depth:
                        depth[nbr] = nd
                        parents[nbr].append(e)
                        q.append(nbr)
        
                    # Alternative parent at the same depth → second shortest path
                    elif depth[nbr] == nd:
                        parents[nbr].append(e)
        
        return sorted(paths_found, key=len)[:k]

    # ---------------------------------------------------------
    # Helper: robust membership retrieval
    # ---------------------------------------------------------
    def members(e, HG):
        try:
            return [str(x) for x in HG.edges[e]]
        except Exception:
            inc = getattr(HG, "incidence_dict", None)
            return [str(x) for x in inc[e]] if inc and e in inc else []

    # ---------------------------------------------------------
    # MAIN LOOP: compute all pairwise shortest paths
    # ---------------------------------------------------------
    all_path_edges = set()
    reports = []

    for u, v in combinations(nodes, 2):
        E_u = node_to_edges.get(u, set())
        E_v = node_to_edges.get(v, set())
        if not E_u or not E_v:
            continue

        edge_paths = shortest_paths_between_edge_sets(E_u, E_v, k=k_paths)
        if not edge_paths:
            continue
        
        for edge_path in edge_paths:
        
            # -------------------
            # Build hop details
            # -------------------
            hops = []
            for Ei, Ej in zip(edge_path[:-1], edge_path[1:]):
                inter_nodes = sorted(set(H.edges[Ei]).intersection(H.edges[Ej]), key=str)
                if len(inter_nodes) < s:
                    warnings.warn(
                        f"Edge hop {Ei}->{Ej} has only {len(inter_nodes)} witness nodes (< s={s})."
                    )
                hops.append({
                    "from_edge": Ei,
                    "to_edge": Ej,
                    "intersection_nodes": [str(x) for x in inter_nodes],
                    "from_members": members(Ei, H),
                    "to_members":   members(Ej, H),
                    "count": len(inter_nodes),
                })
        
            edge_members = {e: members(e, H) for e in edge_path}
        
            reports.append({
                "pair": (u, v),
                "edge_path": edge_path,
                "hops": hops,
                "edge_members": edge_members,
                "start_edge": edge_path[0],
                "end_edge": edge_path[-1],
                "start_comembers": [m for m in edge_members[edge_path[0]] if m != u],
                "end_comembers":   [m for m in edge_members[edge_path[-1]] if m != v],
            })
        
            all_path_edges.update(edge_path)


    # ---------------------------------------------------------
    # Output subgraph
    # ---------------------------------------------------------
    if not all_path_edges:
        return H.restrict_to_edges(set()), []

    H_sub = H.restrict_to_edges(all_path_edges)
    return H_sub, reports




## COLLECT HYPERENTITIES BUT THIS TIME USING SUBDFS: 


import re
import pickle
import pandas as pd

# ==========================================================
# 1) LOAD CHUNK-LEVEL SOURCE/TARGET DATA
# ==========================================================

def load_chunk_dfs(pkl_path):
    import pickle
    import pandas as pd

    with open(pkl_path, "rb") as f:
        df_list = pickle.load(f)

    chunk_to_df = {}

    for df in df_list:
        # Skip non-DF or empty DF
        if not isinstance(df, pd.DataFrame):
            continue
        if df.empty:
            continue
        if "chunk" not in df.columns:
            continue

        # Safe: now at least 1 row exists
        chunk_hash = str(df.iloc[0]["chunk"])
        chunk_to_df[chunk_hash] = df

    return chunk_to_df



# ==========================================================
# 2) EXTRACT DIRECTIONAL SENTENCE FROM AN EDGE ID
# ==========================================================

def generate_directional_sentence(edge_id, chunk_to_df):
    """
    Convert edge_id → directional sentence using EXACT relation label.

    Relation is taken literally from the prefix of the edge_id.
    Format:
        "<target> <relation> <source>."
    """

    m = re.match(r"(.+?)_chunk([0-9A-Za-z]+)_(\d+)", str(edge_id))
    if not m:
        return None

    relation, chunk_hash, row_idx = m.groups()
    row_idx = int(row_idx)

    df = chunk_to_df.get(chunk_hash)
    if df is None or row_idx >= len(df):
        return None

    src_list = df.iloc[row_idx]["source"]
    tgt_list = df.iloc[row_idx]["target"]

    src = ", ".join(map(str, src_list))
    tgt = ", ".join(map(str, tgt_list))

    # Use relation EXACTLY as extracted from the hyperedge label
    rel = relation.strip()

    return f"{src} {rel} {tgt}."



# ==========================================================
# 3) REWRITTEN collect_hyperentities
# ==========================================================

def collect_hyperentities(H_sub, reports, chunk_to_df):
    """
    One directional sentence per hyperedge in correct path order.
    No deduplication. No collapsing.
    """
    sentences = []
    for rpt in reports:
        for e in rpt.get("edge_path", []):
            directional = generate_directional_sentence(e, chunk_to_df)
            if directional:
                sentences.append(directional)
    return sentences

###


def collect_entities(graph):
    nodes = list(graph.nodes)
    edges = list(graph.out_edges(data=True))
    
    try:
        relationships = [ f"{edge[0]} {edge[2]['title']} {edge[1]}." for edge in edges]
    except:
        relationships = [ f"{edge[0]} {edge[2][ list(edges[0][2].keys())[0] ]} {edge[1]}." for edge in edges]

    return " ".join(relationships)


import json


def extract_nodes_from_path_reports(path_reports, lowercase=False, sort=True):
    """
    Return a de-duplicated list of all node labels found anywhere in path_reports.

    Accepts:
      - path_reports: list[dict] as produced by find_shortest_path_hypersubgraph_between_nodes(...)
                      OR a JSON string of that list.

    Options:
      - lowercase: if True, returns all nodes lowercased.
      - sort: if True, returns the list sorted (case-insensitive when lowercase=False).

    Gathers from:
      - report["pair"]
      - report["start_comembers"], report["end_comembers"]
      - report["edge_members"][edge_id] for all edges
      - each hop's "intersection_nodes", "from_members", "to_members"
    """
    # Parse if given as JSON string
    if isinstance(path_reports, str):
        path_reports = json.loads(path_reports)

    nodes = set()

    def _add(items):
        if not items:
            return
        if lowercase:
            for x in items:
                if x is not None:
                    nodes.add(str(x).lower())
        else:
            for x in items:
                if x is not None:
                    nodes.add(str(x))

    if not path_reports:
        return []

    for rpt in path_reports:
        # pair terminals
        _add(rpt.get("pair"))

        # start/end comembers
        _add(rpt.get("start_comembers"))
        _add(rpt.get("end_comembers"))

        # edge_members: dict of edge_id -> [nodes...]
        edge_members = rpt.get("edge_members", {}) or {}
        for _, member_list in edge_members.items():
            _add(member_list)

        # hops: intersection + from/to members
        for hop in rpt.get("hops", []) or []:
            _add(hop.get("intersection_nodes"))
            _add(hop.get("from_members"))
            _add(hop.get("to_members"))

    out = list(nodes)
    if sort:
        out.sort(key=lambda s: s.lower())
    return out

    
# ## FUNCTIONS FOR BETWEENESS CENTRALITY AGENT 

import re, json, ast
import networkx as nx
from itertools import combinations
from collections import defaultdict
from typing import List, Tuple, Union, Dict, Any, Optional

# --- Robust PATH_REPORTS extractor (tolerates code fences & single quotes) ---
def extract_path_reports_from_content(content: str):
    """
    Robustly extract the JSON array that follows 'PATH_REPORTS:' in a string.
    """
    if not content:
        return []

    up = content.upper()
    marker = "PATH_REPORTS:"
    i = up.find(marker)
    if i == -1:
        return []

    j = content.find("[", i)
    if j == -1:
        return []

    depth = 0
    k = j
    in_dq = False
    in_sq = False
    escape = False

    while k < len(content):
        ch = content[k]

        if escape:
            escape = False
            k += 1
            continue

        if ch == "\\":
            escape = True
            k += 1
            continue

        if ch == '"' and not in_sq:
            in_dq = not in_dq
        elif ch == "'" and not in_dq:
            in_sq = not in_sq

        if not in_dq and not in_sq:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    arr_str = content[j:k+1].strip()
                    break
        k += 1
    else:
        return []

    try:
        return json.loads(arr_str)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(arr_str)
            return obj if isinstance(obj, list) else []
        except Exception:
            try:
                arr_str2 = re.sub(r"(?<!\\)'", '"', arr_str)
                return json.loads(arr_str2)
            except Exception:
                return []


def extract_nodes_from_path_reports(path_reports, lowercase: bool = False, sort: bool = True):
    """Collect all unique node labels from path_reports (list or JSON string)."""
    if isinstance(path_reports, str):
        path_reports = json.loads(path_reports)

    nodes = set()

    def _add(items):
        if not items:
            return
        for x in items:
            if x is None:
                continue
            s = str(x)
            nodes.add(s.lower() if lowercase else s)

    for rpt in (path_reports or []):
        _add(rpt.get("pair"))
        _add(rpt.get("start_comembers"))
        _add(rpt.get("end_comembers"))

        edge_members = rpt.get("edge_members") or {}
        for _, member_list in edge_members.items():
            _add(member_list)

        for hop in rpt.get("hops", []) or []:
            _add(hop.get("intersection_nodes"))
            _add(hop.get("from_members"))
            _add(hop.get("to_members"))

    out = list(nodes)
    if sort:
        out.sort(key=str.lower)
    return out

# ------

# --- BAKED-IN: s-betweenness centrality on the s-linegraph of H ---
def s_betweenness_centrality_GLOBAL(
    H,
    s: int = 1,
    edges: bool = False,
    normalized: bool = True,
    return_singletons: bool = True,
) -> Dict[Union[str, int], float]:
    """
    Betweenness centrality on the s-linegraph of a hypergraph H.

    - edges=False → node s-linegraph (two nodes adjacent iff they co-occur in ≥ s hyperedges).
    - edges=True  → edge  s-linegraph (two hyperedges adjacent iff they share ≥ s nodes).
    """
    if s < 1:
        raise ValueError("s must be ≥ 1")

    # Try native HyperNetX linegraph; fall back if missing
    try:
        L = H.get_linegraph(s=s, edges=edges)
    except Exception:
        if not edges:
            # Node s-adjacency via sparse adjacency matrix
            A = H.adjacency_matrix(s=s)                 # node×node
            L = nx.from_scipy_sparse_array(A)
            # map indices to actual node labels
            idx2lab = {i: str(n) for i, n in enumerate(H.nodes)}
            L = nx.relabel_nodes(L, idx2lab, copy=False)
        else:
            # Edge s-linegraph via node→incident-edges index
            L = nx.Graph()
            edge_ids = list(H.edges)
            L.add_nodes_from(edge_ids)
            node_to_edges = defaultdict(set)
            for e in edge_ids:
                for n in H.edges[e]:
                    node_to_edges[str(n)].add(e)
            # overlap counts per edge pair via shared nodes
            overlap = defaultdict(int)
            for inc in node_to_edges.values():
                inc = list(inc)
                for a, b in combinations(inc, 2):
                    if a > b:
                        a, b = b, a
                    overlap[(a, b)] += 1
            for (a, b), cnt in overlap.items():
                if cnt >= s:
                    L.add_edge(a, b)

    if not return_singletons:
        isolates = [v for v, d in L.degree() if d == 0]
        if isolates:
            L = L.copy()
            L.remove_nodes_from(isolates)

    bc = nx.betweenness_centrality(L, normalized=normalized)

    if return_singletons:
        for v in L.nodes:
            bc.setdefault(v, 0.0)

    return bc


# --- Hub connector: find union of shortest s-edge-walks from nodes to the top-betweenness node ---
def find_shortest_between_nodes_and_highbetweeness_node_GLOBAL(
    H,
    nodes: List[Union[str, int]],
    s: int = 1,
    normalized: bool = True,
    return_reports: bool = False,
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """
    1) Compute s-betweenness centrality on the NODE linegraph (edges=False).
    2) Let hub = node with max centrality.
    3) For each input node u, compute the shortest s-edge-walk subgraph between [u, hub]
       via find_shortest_path_hypersubgraph_between_nodes(H, [u, hub], s),
       and union all such path edges.
    4) Return the tight union sub-hypergraph (only those path edges). Optionally return reports.
    """
    if s is None or int(s) < 1:
        raise ValueError("Parameter s must be an integer >= 1")
    s = int(s)

    # 1) highest s-betweenness node
    centrality = s_betweenness_centrality_GLOBAL(
        H, s=s, edges=False, normalized=normalized, return_singletons=True
    )
    if not centrality:
        empty = H.restrict_to_edges(set())
        return (empty, {"hub": None, "hub_centrality": None, "pair_reports": []}) if return_reports else empty

    hub = max(centrality, key=lambda k: centrality[k])
    hub_c = centrality[hub]

    # 2) shortest s-edge-walk from each query node to hub
    query_nodes, seen = [], set()
    for n in nodes or []:
        if n is None:
            continue
        s_n = str(n)
        if s_n == str(hub) or s_n in seen:
            continue
        seen.add(s_n)
        query_nodes.append(s_n)

    union_edges = set()
    all_reports = []

    # NOTE: ensure this function is imported/defined in scope
    # from GraphReasoning import find_shortest_path_hypersubgraph_between_nodes
    for u in query_nodes:
        H_sub_pair, reports = find_shortest_path_hypersubgraph_between_nodes(H, [u, str(hub)], s=s)
        for e in H_sub_pair.edges:
            union_edges.add(e)
        for rpt in (reports or []):
            pair = tuple(map(str, rpt.get("pair", (u, str(hub)))))
            if pair != (u, str(hub)) and pair != (str(hub), u):
                rpt["pair"] = (u, str(hub))
            all_reports.append(rpt)

    if not union_edges:
        empty = H.restrict_to_edges(set())
        return (empty, {"hub": str(hub), "hub_centrality": float(hub_c), "pair_reports": []}) if return_reports else empty

    H_sub_union = H.restrict_to_edges(union_edges)

    if return_reports:
        info = {"hub": str(hub), "hub_centrality": float(hub_c), "pair_reports": all_reports}
        return H_sub_union, info
    return H_sub_union





# tools.py

from typing import List, Union, Tuple, Dict, Any, Optional
from collections.abc import Iterable
from itertools import combinations
from functools import partial

# If these live in another module, import them; otherwise define them before this function.
# from GraphReasoning import s_betweenness_centrality, find_shortest_path_hypersubgraph_between_nodes

def _ensure_s_list(s: Union[int, Iterable]) -> list[int]:
    """Return [s] if s is a scalar; otherwise cast each item to int."""
    if isinstance(s, Iterable) and not isinstance(s, (str, bytes)):
        return [int(x) for x in s]
    return [int(s)]

def _s_centrality_LOCAL(func, H, s=1, edges=False, f=None, return_singletons=True, **kwargs):
    """
    Component-wise s-centrality wrapper.

    - Splits H into s-components (edge or node perspective).
    - Builds the s-linegraph per component.
    - Runs `func(linegraph, **kwargs)` (e.g., nx.betweenness_centrality).
    - Returns {vertex_label: score}, optionally only for `f`.
    """
    # Preferred path: HyperNetX provides s-component subgraphs
    if hasattr(H, "s_component_subgraphs"):
        comps = H.s_component_subgraphs(s=s, edges=edges, return_singletons=return_singletons)
    else:
        # Fallback: build the s-linegraph once and split by connected components
        try:
            L_all = H.get_linegraph(s=s, edges=edges)
        except Exception:
            raise RuntimeError("H must support get_linegraph(s=..., edges=...) or provide s_component_subgraphs().")
        comps = []
        # Build tiny wrapper objects with .nodes/.edges/.shape and get_linegraph returning the CC
        for cc in nx.connected_components(L_all):
            cc_sub = L_all.subgraph(cc).copy()
            # mimic the HNX component API lightly
            class _Comp:
                def __init__(self, L, edges_mode):
                    self._L = L
                    self._edges_mode = edges_mode
                @property
                def nodes(self):  # labels of this component
                    return list(self._L.nodes) if not self._edges_mode else []
                @property
                def edges(self):
                    return list(self._L.nodes) if self._edges_mode else []
                @property
                def shape(self):
                    # HNX uses shape[edges*1] to pick count of vertices in linegraph's domain
                    return (len(self.nodes), len(self.edges))
                def get_linegraph(self, s=1, edges=True):
                    return self._L
            comps.append(_Comp(cc_sub, edges))

    # If a specific node/edge `f` is requested, only keep the component that contains it
    if f is not None:
        for cps in comps:
            if (edges and f in getattr(cps, "edges")) or (not edges and f in getattr(cps, "nodes")):
                comps = [cps]
                break
        else:
            return {f: 0}

    stats = {}
    for h in comps:
        vertices = h.edges if edges else h.nodes
        count = h.shape[1 if edges else 0]
        if count == 1:
            stats.update({v: 0 for v in vertices})
        else:
            g = h.get_linegraph(s=s, edges=edges)
            stats.update(func(g, **kwargs))
        if f is not None:
            # Return only the requested vertex score
            return {f: stats.get(f, 0)}
    return stats


def s_betweenness_centrality_LOCAL(
    H,
    s: Union[int, List[int], Tuple[int, ...], set] = 1,
    edges: bool = False,
    normalized: bool = True,
    return_singletons: bool = True,
) -> Union[Dict[Union[str, int], float], Dict[int, Dict[Union[str, int], float]]]:
    """
    Component-wise s-betweenness using your _s_centrality wrapper.
    - If `s` is a single int: returns {node_or_edge: score}.
    - If `s` is an iterable: returns {s_val: {node_or_edge: score}, ...}.

    Normalization matches your earlier code:
      - Run NX with normalized=False via partial
      - Then apply global factor 2/((n-1)(n-2)) when requested
    """
    s_list = _ensure_s_list(s)
    func = partial(nx.betweenness_centrality, normalized=False)

    def _for_one_s(sv: int) -> Dict[Union[str, int], float]:
        result = _s_centrality_LOCAL(
            func,
            H,
            s=sv,
            edges=edges,
            return_singletons=return_singletons,
        )
        if normalized and H.shape[edges * 1] > 2:
            n = H.shape[edges * 1]
            result = {k: v * 2 / ((n - 1) * (n - 2)) for k, v in result.items()}
        return result

    if len(s_list) == 1:
        return _for_one_s(s_list[0])
    else:
        return {sv: _for_one_s(sv) for sv in s_list}

def find_shortest_between_nodes_and_highbetweeness_node_LOCAL(
    H,
    nodes: List[Union[str, int]],
    s: Union[int, List[int], Tuple[int, ...], set] = 1,
    normalized: bool = True,
    return_reports: bool = False,
):
    """
    For each s in the provided value(s):
      1) Compute s-betweenness on the NODE s-linegraph (component-wise via your centrality helper).
      2) Pick the top node (hub_s).
      3) For each query node u, build a shortest s-edge-walk subgraph between [u, hub_s].
      4) Union all path edges across all s.

    Returns:
      - H_sub (union of path edges), or
      - (H_sub, {"per_s": [...], "pair_reports": [...]}) if return_reports=True
    """
    s_list = _ensure_s_list(s)
    # normalize labels once
    query_nodes = [str(n) for n in (nodes or []) if n is not None]

    union_edges = set()
    all_reports = []
    per_s_meta = []

    for sv in s_list:
        # 1) centrality on node s-linegraph
        cent = s_betweenness_centrality_LOCAL(
            H, s=sv, edges=False, normalized=normalized, return_singletons=True
        )
        if not cent:
            continue
        hub = max(cent, key=lambda k: cent[k])
        hub_c = cent[hub]
        per_s_meta.append({"s": sv, "hub": str(hub), "hub_centrality": float(hub_c)})

        # 2) shortest s-edge-walks from each query node to this hub
        seen = set()
        for u in query_nodes:
            if u == str(hub) or u in seen:
                continue
            seen.add(u)

            H_sub_pair, reports = find_shortest_path_hypersubgraph_between_nodes(
                H, [u, str(hub)], s=sv
            )
            # 3) union edges
            for e in H_sub_pair.edges:
                union_edges.add(e)
            # annotate & collect reports
            for rpt in (reports or []):
                rpt["s"] = sv
                pair = tuple(map(str, rpt.get("pair", (u, str(hub)))))
                if pair != (u, str(hub)) and pair != (str(hub), u):
                    rpt["pair"] = (u, str(hub))
                all_reports.append(rpt)

    if not union_edges:
        empty = H.restrict_to_edges(set())
        return (empty, {"per_s": per_s_meta, "pair_reports": []}) if return_reports else empty

    H_sub_union = H.restrict_to_edges(union_edges)
    if return_reports:
        return H_sub_union, {"per_s": per_s_meta, "pair_reports": all_reports}
    return H_sub_union


### 
#FUNCTIONS FOR CLOSENESS CENTRALITY AGENT


import json, re, ast
from typing import Optional, Any, List, Dict, Tuple, Literal, Union
from collections import defaultdict
from itertools import combinations


# -------------------- s-CLOSENESS centrality (baked-in) --------------------
def s_closeness_centrality_GLOBAL(
    H,
    s: int = 1,
    edges: bool = False,
    wf_improved: bool = True,
    return_singletons: bool = True,
) -> Dict[Union[str, int], float]:
    """
    Closeness centrality on the s-linegraph of a hypergraph H.

    - edges=False → node s-linegraph (two nodes adjacent iff they co-occur in ≥ s hyperedges).
    - edges=True  → edge  s-linegraph (two hyperedges adjacent iff they share ≥ s nodes).
    Uses networkx.closeness_centrality (wf_improved=True by default).
    """
    if s < 1:
        raise ValueError("s must be ≥ 1")

    # Build the s-linegraph (prefer native H.get_linegraph; fallback if needed)
    try:
        L = H.get_linegraph(s=s, edges=edges)
    except Exception:
        if not edges:
            # Node s-adjacency via sparse adjacency matrix
            A = H.adjacency_matrix(s=s)  # node×node
            L = nx.from_scipy_sparse_array(A)
            idx2lab = {i: str(n) for i, n in enumerate(H.nodes)}
            L = nx.relabel_nodes(L, idx2lab, copy=False)
        else:
            # Edge s-linegraph via node→incident-edges index
            L = nx.Graph()
            edge_ids = list(H.edges)
            L.add_nodes_from(edge_ids)
            node_to_edges = defaultdict(set)
            for e in edge_ids:
                for n in H.edges[e]:
                    node_to_edges[str(n)].add(e)
            overlap = defaultdict(int)
            for inc in node_to_edges.values():
                inc = list(inc)
                for a, b in combinations(inc, 2):
                    if a > b: a, b = b, a
                    overlap[(a, b)] += 1
            for (a, b), cnt in overlap.items():
                if cnt >= s:
                    L.add_edge(a, b)

    # Optionally drop isolates before computing
    if not return_singletons:
        isolates = [v for v, d in L.degree() if d == 0]
        if isolates:
            L = L.copy()
            L.remove_nodes_from(isolates)

    # NetworkX closeness (handles disconnected by reachable fraction)
    cc = nx.closeness_centrality(L, wf_improved=wf_improved)

    # Ensure isolates appear with 0 if we kept them
    if return_singletons:
        for v in L.nodes:
            cc.setdefault(v, 0.0)

    return cc

# ----- Build union of shortest s-edge-walks from nodes to the top-closeness node -----
def find_shortest_between_nodes_and_highcloseness_node_GLOBAL(
    H,
    nodes: List[Union[str, int]],
    s: int = 1,
    wf_improved: bool = True,
    return_reports: bool = False,
):
    """
    1) Compute s-closeness centrality on the NODE linegraph (edges=False).
    2) Let hub = node with max closeness.
    3) For each input node u, compute the shortest s-edge-walk subgraph between [u, hub]
       via find_shortest_path_hypersubgraph_between_nodes(H, [u, hub], s),
       and union all such path edges.
    4) Return the tight union sub-hypergraph (only those path edges). Optionally return reports.
    """
    if s is None or int(s) < 1:
        raise ValueError("Parameter s must be an integer >= 1")
    s = int(s)

    # 1) highest s-closeness node (on node s-linegraph)
    closeness = s_closeness_centrality_GLOBAL(
        H, s=s, edges=False, wf_improved=wf_improved, return_singletons=True
    )
    if not closeness:
        empty = H.restrict_to_edges(set())
        return (empty, {"hub": None, "hub_closeness": None, "pair_reports": []}) if return_reports else empty

    hub = max(closeness, key=lambda k: closeness[k])
    hub_c = closeness[hub]

    # 2) shortest s-edge-walk from each query node to hub
    query_nodes, seen = [], set()
    for n in nodes or []:
        if n is None: continue
        s_n = str(n)
        if s_n == str(hub) or s_n in seen: continue
        seen.add(s_n); query_nodes.append(s_n)

    union_edges = set()
    all_reports = []
    for u in query_nodes:
        H_sub_pair, reports = find_shortest_path_hypersubgraph_between_nodes(H, [u, str(hub)], s=s)
        for e in H_sub_pair.edges:
            union_edges.add(e)
        for rpt in (reports or []):
            pair = tuple(map(str, rpt.get("pair", (u, str(hub)))))
            if pair != (u, str(hub)) and pair != (str(hub), u):
                rpt["pair"] = (u, str(hub))
            all_reports.append(rpt)

    if not union_edges:
        empty = H.restrict_to_edges(set())
        return (empty, {"hub": str(hub), "hub_closeness": float(hub_c), "pair_reports": []}) if return_reports else empty

    H_sub_union = H.restrict_to_edges(union_edges)

    if return_reports:
        info = {"hub": str(hub), "hub_closeness": float(hub_c), "pair_reports": all_reports}
        return H_sub_union, info
    return H_sub_union

###


####



def detect_communities(graph):    
    communities = community_louvain.best_partition(graph)
    return communities

def summarize_communities(graph, communities, generate):
    community_summaries = []
    for index, community in tqdm(enumerate(communities)):
        description = collect_entities(graph.subgraph(community))
#         nodes = list(subgraph.nodes)
#         edges = list(subgraph.out_edges(data=True))
#         description = "Relationships: "
#         relationships = []
#         for edge in edges:
#             relationships.append(
#                 f"{edge[0]} {edge[2]['title']} {edge[1]}.")
            
#         description += " ".join(relationships)

        response = generate(system_prompt= "You are an expert in multiple engineering fields. Summarize the following relationships and make a professional report.",
                                       prompt= description)

        print(description)
        summary = response.strip()
        community_summaries.append(summary)
    return community_summaries

EXTRACT_PROMPT = """
You are a strict scientific keyword extractor.

Your job is to extract ONLY the concrete scientific entities mentioned
in the text (e.g., materials, chemicals, biological entities, properties).
Do NOT extract abstract concepts, verbs, or relational words.

Rules:
- Output ONLY valid JSON.
- Format: {"keywords": ["keyword1", "keyword2", ...]}
- Extract ONLY MATERIALS / SUBSTANCES / SPECIFIC ENTITIES.
- DO NOT extract:
    - verbs (e.g., relate, interact, form, behave)
    - question words (how, why, what)
    - abstract concepts (mechanistic relation, mechanism, relationship)
    - adjectives or descriptors (mechanistic, structural, functional)
- Keep acronyms in original case (e.g., PCL, PLA, PEG).
- Otherwise, lowercase all extracted words.
- No explanations, no markdown, no code fences.

Examples:

Context: What is the capital of the United States?
{"keywords": ["united states"]}

Context: How can silk mechanistically relate to PCL?
{"keywords": ["silk", "PCL"]}

Context: What technology is Taiwan famous for?
{"keywords": ["taiwan", "technology", "semiconductor"]}

Context: What is CVD uniformity and etching uniformity?
{"keywords": ["cvd", "etching", "uniformity"]}
"""

##TEST
import re
import json

def safe_parse_json(response):
    """
    Safely parse JSON from LLM output, repairing common issues:
    - Extraneous text before/after JSON
    - Newlines, spaces, markdown fences
    - Single quotes instead of double quotes
    - Raw lists (["x", "y"]) instead of {"keywords": [...]}
    """
    if not response or not response.strip():
        raise ValueError("LLM returned an empty response.")

    raw = response.strip()

    # Remove code fences if they appear
    raw = raw.replace("```json", "").replace("```", "").strip()

    # First: try direct JSON
    try:
        return json.loads(raw)
    except:
        pass

    # Try to extract JSON object {...}
    match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except:
            pass

    # Try converting single quotes → double quotes
    candidate2 = raw.replace("'", '"')
    try:
        return json.loads(candidate2)
    except:
        pass

    # Try converting raw list → JSON object
    if raw.startswith("[") and raw.endswith("]"):
        try:
            lst = json.loads(raw)
            return {"keywords": lst}
        except:
            pass

    raise ValueError(f"Could not parse JSON from LLM output:\n{response}")


# EXTRACT KEYWORDS AND GET THE node degree 

def get_node_degree(H, node):
    """Return the degree of a node in a HyperNetX hypergraph."""
    count = 0
    node = str(node)
    for e in H.edges:
        if node in H.edges[e]:
            count += 1
    return count


def extract_keywords_to_nodes(
    question, generate, node_embeddings, embedding_tokenizer, embedding_model,
    N_samples=5, similarity_threshold=0.9, H=None
):
    # Ask model
    response = generate(system_prompt=EXTRACT_PROMPT,
                        prompt=f"Context: {question}")

    print("\nRAW LLM OUTPUT:", response)

    # Clean + parse
    parsed = safe_parse_json(response)

    keywords = parsed.get("keywords", [])
    print(f"Extracted keywords: {keywords}")

    # Find best-fitting graph nodes
    elements = [
        np.array(find_best_fitting_node_list(
            keyword, node_embeddings, embedding_tokenizer,
            embedding_model, N_samples=N_samples,
            similarity_threshold=similarity_threshold
        ))[:, 0]
        for keyword in keywords
    ]

    nodes = [elem for arr in elements for elem in arr]
    print(f"Found matched nodes in embeddings: {nodes}")

    # # ----------------------------------------------------
    # # New! Compute node degrees as an intermediate check
    # # ----------------------------------------------------
    # if H is not None:
    #     print("\n=== NODE DEGREE CHECK ===")
    #     for n in nodes:
    #         deg = get_node_degree(H, n)
    #         print(f"Node '{n}' → degree {deg}")
    #     print("==========================\n")

    return nodes


EXTRACT_MATERIAL_PROMPT = """
You are a strict keyword extractor.

Rules:
- Output EXACTLY one JSON object: {"keywords": [<strings>]} with no extra text.
- If any materials/chemicals/compounds are present, RETURN ONLY those (lowercased, deduplicated).
- Otherwise, include domain nouns (processes, properties, entities), but never verbs, stopwords, or question words.
- Preserve common acronyms (e.g., CVD, PLA) in their original case; otherwise lowercase.
- No explanations.

Example:
Context: ```What is a formulation for a composite design that can combine chitosan and silk?```
{"keywords": ["chitosan", "silk"]}
"""

def extract_material_keywords_to_nodes(question, generate, node_embeddings, embedding_tokenizer, embedding_model, N_samples=5, similarity_threshold=0.9):
    response = generate(system_prompt=EXTRACT_MATERIAL_PROMPT,
                        prompt=f'Context: ```{question}```')
    # keywords = response.replace('[',' ').replace(']',' ').replace(',',' ').split()
    # print(f'Extracted {keywords} in {question}')
    
    response = extract(response)
    keywords = json.loads(response)
    print(f'Extract keywords: {keywords}')
    elements = [ np.array(find_best_fitting_node_list(keyword, node_embeddings, embedding_tokenizer, embedding_model, N_samples=N_samples, similarity_threshold=similarity_threshold))[:, 0] for keyword in keywords]

    nodes = []
    for element in elements:
        nodes += list(element)
    print(f'Found {nodes} in node_embeddings')

    return nodes

    

def local_search(question, generate, graph, node_embeddings, embedding_tokenizer, embedding_model, N_samples=5, similarity_threshold=0.9):
    
    nodes = extract_keywords_to_nodes(question, generate, node_embeddings, embedding_tokenizer, embedding_model, N_samples, similarity_threshold)
    #------ local search on the subgraph from shortest path search

    subgraph = find_shortest_path_subgraph_between_nodes(graph.to_directed(), nodes)
    information = collect_entities(subgraph)

    response = generate(system_prompt= "Answer the query detailedly based on the collected information and the provided current thought. If you think the report doesn't help, you should just keep the current thought.",
                               prompt=f"Based on the following... Report: {information}. I can give you the detailed answer to the query: {question}")
    #--- validation ---
    reason = generate(system_prompt= "You are a senior professional in the field. Answer yes or no whether the answer is good enough for the question. You only provide reason when you think it is not answering the question.",
                               prompt=f"Question: {question} Answer: {response}")
    if 'yes' in reason.lower():
        return response
    else:
        return reason

def global_search(question, generate, graph, communities, community_summaries, node_embeddings, embedding_tokenizer, embedding_model, N_samples=5, similarity_threshold=0.9):
    
    nodes = extract_keywords_to_nodes(question, generate, node_embeddings, embedding_tokenizer, embedding_model, N_samples, similarity_threshold)

    #------ global search on the subgraph from communities
    subgraph = find_shortest_path_subgraph_between_nodes(graph.to_directed(), nodes)
    information = collect_entities(subgraph)

    target_community=set()
    for i, community in enumerate(communities):
        for node in nodes:
            if node in community:
                target_community.add(i)
    print(target_community)

    last_response=''
    # all_responses=[]    
    for i, summary in enumerate(np.array(community_summaries)[list(target_community)]):
        response = generate(system_prompt= "Answer the query detailedly based on the collected information and the provided current thought. If you think the report doesn't help, you should just keep the current thought.",
                                   prompt=f"Based on the following... Report:{summary}. Information: {information}. Current thought: {last_response}. I can give you the detailed answer to the query: {question}")
    #     all_responses.append(response)
        last_response=response
        
    # response = generate(system_prompt= "Answer the query detailedly based on the collected information and the provided current thought.",
    #                                prompt=f"Based on the following... Previous thoughts:{' '.join(all_responses)}. Information: {information}. I can give you the detailed answer to the query: {question}")
     
    
    return response