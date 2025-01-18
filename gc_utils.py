import sklearn
import numpy as np
import pandas as pd
import torch
import networkx as nx
from numpy.array_api import argmax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA

from scipy.stats import percentileofscore

import plotly.express as px
from scipy.stats import mode
from sklearn.metrics import pairwise_distances

import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.units import degrees
from sympy.stats.sampling.sample_numpy import numpy


def is_k_color(adj, k_assignment, print_bad=False):
    """
    Check if the graph is a valid k-coloring.
    """
    disagreement_count = 0
    adj_clone = adj.clone().detach().cpu().numpy()
    n = adj_clone.shape[0]
    k_assignment += 1
    # for i in range(n):
    #     adj_clone[i, adj_clone[i] > 0] = k_assignment[i] + 1
    bad_indexes = []
    for i in range(n):
        neighbors_assignment = k_assignment[adj_clone[i]!=0]
        bad_count = (neighbors_assignment == k_assignment[i]).sum()
        if bad_count > 0:
            bad_indexes.append(i)
            if print_bad:
                print(i, bad_count, np.arange(n)[adj_clone[i] != 0][neighbors_assignment == k_assignment[i]])
            # disagreement_count += bad_count
            disagreement_count += 1
    # return disagreement_count == 0, disagreement_count//2, bad_indexes
    return disagreement_count == 0, disagreement_count, bad_indexes



from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import numpy as np

def sklearn_k_means_3(adj, features, k=3, max_iters=100):
    # Ensure features is a numpy array for sklearn compatibility
    features_np = features.clone().detach().cpu().numpy()
    adj_np = adj.clone().detach().cpu().numpy()

    # Perform initial k-means clustering
    kmeans = KMeans(n_clusters=k, max_iter=max_iters)
    kmeans.fit(features_np)
    assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_np)

    # Find connected components
    connected_components = list(nx.connected_components(G))

    # Reassign clusters in each connected component
    for component in connected_components:
        component = sorted(component)  # Sort to get the lowest index node first
        # lowest_index_node = component[0]
        degrees = [G.degree(node) for node in component]
        largest_degree_node = component[np.argmax(degrees)]

        current_cluster = assignments[largest_degree_node]

        # If the current cluster of the lowest index node is not 0, reassign
        if current_cluster != 0:
            # Find nodes currently assigned to cluster 0
            nodes_in_cluster_0 = np.where(assignments == 0)[0]

            # Swap the clusters
            assignments[assignments == current_cluster] = -1  # Temporary assignment
            assignments[nodes_in_cluster_0] = current_cluster
            assignments[assignments == -1] = 0

    # Recompute centroids based on the new assignments
    new_centroids = np.array([features_np[assignments == i].mean(axis=0) for i in range(k)])

    return assignments, new_centroids

def sklearn_k_means(X, k, max_iters=100, centroids = None):
    """
    Perform k-means clustering on a dataset using scikit-learn and reorder the clusters based on PCA.

    Parameters:
    - X: A torch tensor of shape (n, d) where n is the number of data points and d is the dimensionality.
    - k: The number of clusters.
    - max_iters: Maximum number of iterations to run the algorithm.

    Returns:
    - cluster_assignments: An array of reordered cluster assignments for each data point.
    - centroids: The reordered centroids of the clusters.
    """
    # Ensure X is a numpy array for sklearn compatibility
    X_np = X.clone().detach().cpu().numpy() if type(X) == torch.Tensor else X
    pca = PCA(n_components=2)
    pca.fit(X_np)

    if centroids is not None:
        distances = np.linalg.norm(X_np[:, np.newaxis] - centroids, axis=2)
        assignments = np.argmin(distances, axis=1)
        return assignments, centroids


    X_np_unique = np.unique(X_np, axis=0)
    k = min(k, X_np_unique.shape[0])
    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=k, max_iter=max_iters)
    kmeans.fit(X_np)

    # Retrieve the cluster assignments and centroids
    cluster_assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Apply PCA to reduce centroids to 2 dimensions
    centroids_2d = (centroids - pca.mean_) @ pca.components_.T

    # Identify the most right centroid
    most_right_index = np.argmax(centroids_2d[:, 0])

    # Calculate angles for the remaining centroids
    angles = np.arctan2(centroids_2d[:, 1], centroids_2d[:, 0])
    angles = (angles - angles[most_right_index]) % (2 * np.pi)

    # Sort indices based on angles in a clockwise manner
    sorted_indices = [most_right_index] + sorted(
        [i for i in range(k) if i != most_right_index],
        key=lambda i: angles[i]
    )

    # Create a mapping from old cluster indices to new ones
    new_cluster_map = {old: new for new, old in enumerate(sorted_indices)}

    # Reassign cluster labels based on the sorted centroids
    new_cluster_assignments = [new_cluster_map[label] for label in cluster_assignments]
    new_cluster_assignments = np.array(new_cluster_assignments)
    new_centroids = centroids[sorted_indices]

    return new_cluster_assignments, new_centroids

def sklearn_k_means2(X, k, max_iters=100):
    """
    Perform k-means clustering on a dataset using scikit-learn.

    Parameters:
    - X: A torch tensor of shape (n, d) where n is the number of data points and d is the dimensionality.
    - k: The number of clusters.
    - max_iters: Maximum number of iterations to run the algorithm.

    Returns:
    - cluster_assignments: An array of cluster assignments for each data point.
    - centroids: The final centroids of the clusters.
    """
    # Ensure X is a numpy array for sklearn compatibility
    X_np = X.clone().detach().cpu().numpy()

    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=k, max_iter=max_iters)
    kmeans.fit(X_np)

    # Retrieve the cluster assignments and centroids
    cluster_assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return cluster_assignments, centroids


def reassign_clusters_respecting_order(X, adj, k, max_iters=100):
    X = X.clone().detach().cpu().numpy()
    adj = adj.clone().detach().cpu().numpy()
    # Perform initial k-means clustering
    kmeans = KMeans(n_clusters=k, max_iter=max_iters)
    kmeans.fit(X)
    assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Calculate distances of nodes from their centroids
    distances = np.linalg.norm(X - centroids[assignments], axis=1)

    # Sort nodes by their distance from their centroid
    sorted_nodes = np.argsort(distances)

    visited_nodes = set()
    for node in sorted_nodes:
        neighbors = np.where(adj[node] > 0)[0]
        conflict_neighbors = [neighbor for neighbor in neighbors if neighbor in visited_nodes and assignments[neighbor] == assignments[node]]

        if conflict_neighbors:
            # Find the closest centroid that is not the current one and is allowed
            allowed_centroids = [i for i in range(k) if i != assignments[node] and all(assignments[neighbor] != i for neighbor in visited_nodes)]
            min_distance = np.inf
            new_cluster = assignments[node]

            for centroid_index in allowed_centroids:
                distance = np.linalg.norm(X[node] - centroids[centroid_index])
                if distance < min_distance:
                    min_distance = distance
                    new_cluster = centroid_index

            # Reassign the node to the new cluster
            assignments[node] = new_cluster

        visited_nodes.add(node)

    return assignments

from ortools.sat.python import cp_model

def find_closest_kcoloring(assignment, adj_matrix, k, hint = None):
    assignment = [a-1 for a in assignment]

    model = cp_model.CpModel()
    n = len(adj_matrix)  # Number of nodes
    colors = [model.NewIntVar(0, k-1, f'node_{i}') for i in range(n)]

    # Add adjacency constraints
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1:
                model.Add(colors[i] != colors[j])
    if hint is not None:
        for i in range(n):
            if hint[i] != -1:
                model.Add(colors[i] == hint[i]-1)


    # Objective: minimize the sum of absolute differences from the initial assignment
    objective_terms = []
    for i in range(n):
        if assignment[i] < 0: continue
        deviation = model.NewIntVar(-k, k, f'deviation_{i}')
        model.Add(deviation == colors[i] - assignment[i])
        abs_deviation = model.NewIntVar(0, k, f'abs_deviation_{i}')
        model.AddAbsEquality(abs_deviation, deviation)
        objective_terms.append(abs_deviation)

    model.Minimize(sum(objective_terms))

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return [solver.Value(colors[i])+1 for i in range(n)]
    else:
        return None  # No solution found

def find_all_k_coloring(adj_matrix, k):
    model = cp_model.CpModel()
    n = len(adj_matrix)  # Number of nodes
    colors = [model.NewIntVar(0, k-1, f'node_{i}') for i in range(n)]

    # Add adjacency constraints
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1:
                model.Add(colors[i] != colors[j])

    # Create a solution collector
    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self, colors):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.colors = colors
            self.solutions = []

        def on_solution_callback(self):
            self.solutions.append([self.Value(color) + 1 for color in self.colors])

    # Solve the model
    solver = cp_model.CpSolver()
    solution_collector = SolutionCollector(colors)
    status = solver.SearchForAllSolutions(model, solution_collector)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return solution_collector.solutions
    else:
        return None  # No solution found

def find_k_coloring(adj_matrix, k):
    model = cp_model.CpModel()
    n = len(adj_matrix)  # Number of nodes
    colors = [model.NewIntVar(0, k-1, f'node_{i}') for i in range(n)]

    # Add adjacency constraints
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1:
                model.Add(colors[i] != colors[j])

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return [solver.Value(colors[i])+1 for i in range(n)]
    else:
        return None  # No solution found

def check_k_colorable_and_assign(adj_matrix, k):
    model = cp_model.CpModel()
    n = len(adj_matrix)  # Number of nodes
    # Create a color variable for each node with domain [0, k-1]
    colors = [model.NewIntVar(0, k - 1, f'node_{i}') for i in range(n)]

    # Add constraints: no two adjacent nodes can have the same color
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1:
                model.Add(colors[i] != colors[j])

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        # If the problem is solvable, extract the color assignment
        color_assignment = [solver.Value(colors[i]) for i in range(n)]
        return True, color_assignment
    else:
        # If the problem is not solvable, return False with an empty assignment
        return False, []


def plot(model,data):
    from sklearn.decomposition import PCA
    import plotly.express as px

    k = len(data["color"]["batch"]) + (data.label[0].item()==0)
    n = len(data["vertex"]["batch"])
    edges = data.edge_index_dict[('vertex', 'connected_to', 'vertex')]
    adj = torch.zeros((n, n))
    adj[edges[0], edges[1]] = 1
    adj[edges[1], edges[0]] = 1
    ass, cent = sklearn_k_means(model.history, k)
    is_k_col = is_k_color(adj, ass)
    dist = 0
    markers = [0]*n
    if is_k_col is False:
        close_ass= find_closest_kcoloring(ass, adj, k)
        markers = [int(a!=c) for a,c in zip(ass, close_ass)]
        dist = sum(markers)
        print(dist)
        return

    features = model.history.clone().detach().cpu().numpy()
    pca = PCA(2)
    lower = pca.fit_transform(features)
    x = lower[:, 0]
    y = lower[:, 1]
    fig = px.scatter(x=x, y=y, title=f'Scatter Plot of x vs. y nodes {is_k_col} {dist}', labels={'x': 'x', 'y': 'y'}, color=[f"{ass[i]}" for i in range(n)])
    fig.show()
    if is_k_col is False:
        fig = px.scatter(x=x, y=y, title=f'Scatter Plot of x vs. y nodes closest assignment',
                         labels={'x': 'x', 'y': 'y'},
                         color=[f"{close_ass[i]}" for i in range(n)])
        fig.show()
        fig = px.scatter(x=x, y=y, title=f'Scatter Plot of x vs. y nodes closest assignment diff', labels={'x': 'x', 'y': 'y'},
                         color=[f"{m}" for m in markers])
        fig.show()
    print()
    # features = model.history_c.clone().detach().cpu().numpy()
    # ass, cent = sklearn_k_means(model.history_c, k)
    # is_k_col = is_k_color(adj, ass)
    # pca = PCA(2)
    # lower = pca.fit_transform(features)
    # x = lower[:, 0]
    # y = lower[:, 1]
    # fig = px.scatter(x=x, y=y, title=f'Scatter Plot of x vs. y color {is_k_col}', labels={'x': 'x', 'y': 'y'})
    # fig.show()


def attributes_plot(adj, ass, features, k):
    adj = adj.clone().detach().cpu().numpy()
    features_orig = features.clone().detach().cpu().numpy()
    for color in range(1,k+1):
        curr_indexes = ass==color
        features = features_orig[curr_indexes]

        pca = KernelPCA(n_components=3, kernel='rbf')
        lower = pca.fit_transform(features)
        x = lower[:, 0]
        y = lower[:, 1]
        z = lower[:, 2]
        # Calculate degrees
        degrees = adj[:,curr_indexes][curr_indexes,:].sum(axis=0)

        # Chromatic number is not calculated per node. It's a property of the graph.

        # Number of available colors for each node (simplified version)
        # Assuming a fixed set of colors represented by integers [0, k-1]
        available_colors = [k - len(set(ass[adj[node] > 0])) for node in range(len(ass))]

        # Number of conflicts for each node
        conflicts = [sum(ass[neighbor] == ass[node] for neighbor in np.where(adj[node] > 0)[0]) for node in range(len(ass))]

        # Number of conflicts if the color were to change (simplified version)
        # This is a more complex calculation, here's a basic approach
        conflicts_if_change = []
        for node in range(len(ass)):
            current_conflicts = conflicts[node]
            conflict_changes = []
            for new_color in range(k):
                if new_color != ass[node]:
                    new_conflicts = sum(ass[neighbor] == new_color for neighbor in np.where(adj[node] > 0)[0])
                    conflict_changes.append(new_conflicts)
                # else:
                #     conflict_changes.append(current_conflicts)
            conflicts_if_change.append(np.mean(conflict_changes) if len(conflict_changes)>0 else 0)

        # Now, you can use these lists to color your plot. Here's an example using degrees:
        fig = px.scatter(x=x, y=y, title='Scatter Plot colored by degree',
                         labels={'x': 'x', 'y': 'y'},
                         color=degrees)
        fig.show()



def calculate_sum_and_avg_degree_of_neighbors(adj):
    # Calculate the degree of each node (sum of connections)
    degrees = adj.sum(dim=1)

    # Calculate the sum of degrees of neighbors for each node
    neighbor_degrees_sum = torch.matmul(adj, degrees)

    # Count the number of neighbors for each node
    neighbor_counts = degrees

    # Avoid division by zero for isolated nodes by replacing 0s with 1s temporarily for division
    neighbor_counts = neighbor_counts.masked_fill(neighbor_counts == 0, 1)

    # Calculate the average degree of neighbors
    avg_degrees = neighbor_degrees_sum / neighbor_counts

    # Reset the average degree of isolated nodes to 0
    avg_degrees = avg_degrees.masked_fill(neighbor_counts == 1, 0)

    return neighbor_degrees_sum, avg_degrees


def count_neighbors_colors(adj, assignment):
    n = adj.size(0)  # Number of nodes
    k = torch.max(assignment) + 1  # Assuming colors are 0-indexed and contiguous
    neighbor_color_counts = torch.zeros((n, k), dtype=torch.int64)

    for color in range(k):
        color_mask = (assignment == color).float()  # Create a mask for nodes of this color
        neighbor_colors = torch.matmul(adj, color_mask.unsqueeze(-1)).squeeze()  # Count neighbors of this color
        neighbor_color_counts[:, color] = neighbor_colors

    return neighbor_color_counts

def calculate_avg_distance(features, adj):
    # Calculate pairwise squared Euclidean distances
    dist_squared = torch.sum(features ** 2, dim=1, keepdim=True) + torch.sum(features ** 2, dim=1) - 2 * torch.matmul(
        features, features.t())
    dist = torch.sqrt(torch.relu(dist_squared))  # Ensure non-negative before sqrt; relu for numerical stability

    # Use adjacency matrix to mask distances between non-neighbors
    neighbor_mask = adj > 0  # Assuming adjacency matrix is binary (1 for neighbors, 0 otherwise)
    masked_distances = dist * neighbor_mask.float()

    # Calculate the sum of distances and the number of neighbors for each node
    sum_distances = torch.sum(masked_distances, dim=1)
    num_neighbors = torch.sum(neighbor_mask, dim=1)

    # Avoid division by zero for isolated nodes by replacing 0 with 1 temporarily
    num_neighbors = torch.where(num_neighbors == 0, torch.ones_like(num_neighbors), num_neighbors)

    # Calculate average distance
    avg_distance = sum_distances / num_neighbors

    # Optionally, handle isolated nodes by setting their avg distance to a specific value (e.g., 0)
    avg_distance = torch.where(num_neighbors == 1, torch.zeros_like(avg_distance),
                               avg_distance)  # Reset isolated nodes to 0

    return avg_distance




def leave_one_out_knn(features, assignments, k=3):
    """
    Reassign each assignment using leave-one-out KNN.

    :param features: numpy array of shape (n_samples, n_features)
    :param assignments: numpy array of shape (n_samples,) with original assignments
    :param k: number of nearest neighbors to consider
    :return: numpy array of shape (n_samples,) with reassigned labels
    """
    n_samples = features.shape[0]
    reassigned_labels = np.zeros(n_samples)

    # Calculate pairwise distances between all samples
    distances = pairwise_distances(features, metric='euclidean')

    # Inside the loop of leave_one_out_knn function
    for i in range(n_samples):
        # Exclude the current sample from distance calculation
        distances[i, i] = np.inf

        # Find the indices of the k nearest neighbors
        neighbors_idx = np.argsort(distances[i, :])[:k]

        # Assign the current sample to the majority class among its neighbors
        neighbor_labels = assignments[neighbors_idx]
        mode_result = mode(neighbor_labels)
        # Check if mode_result.mode is scalar and handle accordingly
        if np.isscalar(mode_result.mode):
            reassigned_labels[i] = mode_result.mode
        else:
            reassigned_labels[i] = mode_result.mode[0]
    return reassigned_labels

















def initialize_nodes(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])
    return positions


def classify_third(angle):
    if 0 <= angle < 2 * np.pi / 3:
        return 0
    elif 2 * np.pi / 3 <= angle < 4 * np.pi / 3:
        return 1
    else:
        return 2


def adjust_positions(positions, adj_matrix, iterations=100, learning_rate_away=0.05, learning_rate_close=0.005):
    n = len(positions)
    for _ in range(iterations):
        new_positions = positions.copy()
        for i in range(n):
            neighbors = np.where(adj_matrix[i] == 1)[0]
            if len(neighbors) > 0:
                mean_neighbor_position = np.mean(positions[neighbors], axis=0)
                direction = positions[i] - mean_neighbor_position
                distance = np.linalg.norm(direction)
                if distance == 0:
                    direction = np.random.randn(2)  # Random direction if distance is zero
                    distance = np.linalg.norm(direction)
                new_positions[i] += learning_rate_away * direction / distance  # Move away from neighbors

            angle = np.arctan2(positions[i][1], positions[i][0])
            third = classify_third(angle)
            same_third_nodes = [j for j in range(n) if classify_third(np.arctan2(positions[j][1], positions[j][0])) == third and j not in neighbors]
            for node in same_third_nodes:
                if node != i:
                    direction = positions[node] - positions[i]
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        new_positions[i] -= learning_rate_close * direction / distance  # Move closer to nodes in the same third

            if np.linalg.norm(new_positions[i]) > 1:
                new_positions[i] = new_positions[i] / np.linalg.norm(new_positions[i])  # Clamp to unit circle

        positions = new_positions
    return positions
def plot_positions(positions):
    plt.figure(figsize=(6, 6))
    plt.scatter(positions[:, 0], positions[:, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def max_distance_between_adjacent_nodes(features, adj):
    # Normalize the features tensor
    # features = features / torch.norm(features, dim=1, keepdim=True)

    min_distance = np.inf
    num_nodes = adj.shape[0]

    for i in range(num_nodes):
        # Find indices of adjacent nodes
        adjacent_nodes = torch.where(adj[i] == 1)[0]

        if len(adjacent_nodes) > 0:
            # Calculate dot products with adjacent nodes
            dot_products = torch.matmul(features[i], features[adjacent_nodes].T)
            distances = dot_products
            # Normalize the dot products
            # norms_i = torch.norm(features[i])
            # norms_adj = torch.norm(features[adjacent_nodes], dim=1)
            # distances = dot_products / (norms_i * norms_adj)
            min_distance = min(min_distance, distances.max().item())

    return min_distance

def lovasz_theta_max_dist(adj, features):
    t = max_distance_between_adjacent_nodes(features, adj)
    return lovasz_theta(t)

def lovasz_theta(t):
    return 1-1/t if t != 0 else -1



def most_frequent_number(vector):
    vector = torch.tensor(vector) if not isinstance(vector, torch.Tensor) else vector
    unique_numbers, counts = torch.unique(vector, return_counts=True)
    return unique_numbers[torch.argmax(counts)].item()


import numpy as np
from ortools.linear_solver import pywraplp
import cvxpy as cp


def generate_random_unit_vector(n):
    vec = np.random.normal(0, 1, n)
    return vec / np.linalg.norm(vec)


def sdp_coloring(adj_matrix):
    num_vertices = len(adj_matrix)
    n = num_vertices

    # Step 1: Initialize vectors for each vertex
    vectors = {v: generate_random_unit_vector(n) for v in range(num_vertices)}

    # Step 2: SDP constraints
    X = cp.Variable((n, n), symmetric=True)
    t = cp.Variable()
    constraints = [X >> 0]  # X is positive semidefinite

    for i in range(n):
        constraints.append(X[i, i] == 1)  # Unit length constraint

    for u in range(num_vertices):
        for v in range(num_vertices):
            if adj_matrix[u][v] == 1:
                constraints.append(X[u, v] <= t)  # Distance constraint for adjacent nodes

    # Objective function: minimize the maximum distance t
    objective = cp.Minimize(t)

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Extract the coloring and vectors
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        colors = {}
        for i in range(num_vertices):
            colors[i] = np.argmax(X.value[i])

        # Extract eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(X.value)
        node_vectors = eigenvectors[:, -n:]  # Take the eigenvectors corresponding to the largest eigenvalues

        # print("Value of t:", t.value)
        # print("Lovasz theta:", lovasz_theta(t.value))
        return lovasz_theta(t.value), colors, node_vectors
    else:
        # print('The problem does not have an optimal solution.')
        return None, None, None


def greedyColoring(adj, V):
    result = [-1] * V
    result[0] = 0  # Assign the first color to the first vertex

    available = [False] * V  # Temporary array to store the available colors

    for u in range(1, V):
        # Process all adjacent vertices and flag their colors as unavailable
        for i in range(V):
            if adj[u, i] == 1 and result[i] != -1:
                available[result[i]] = True

        # Find the first available color
        cr = 0
        while cr < V:
            if not available[cr]:
                break
            cr += 1

        # Assign the found color
        result[u] = cr

        # Reset the values back to false for the next iteration
        for i in range(V):
            if adj[u, i] == 1 and result[i] != -1:
                available[result[i]] = False

    return len(set(result)), result




def count_conflicts(adj_matrix, assignment, num_nodes, node, color):
    return sum(1 for neighbor in range(num_nodes) if adj_matrix[node, neighbor] == 1 and assignment[neighbor] == color)

def least_conflicting_color(adj_matrix, assignment, num_nodes, node, k):
    current_color = assignment[node]
    min_conflicts = count_conflicts(adj_matrix, assignment, num_nodes, node, current_color)
    best_color = current_color
    for color in range(k):
        conflicts = count_conflicts(adj_matrix, assignment, num_nodes, node, color)
        if conflicts < min_conflicts:
            min_conflicts = conflicts
            best_color = color
    return best_color
def greedy_coloring_least_conflicts(adj_matrix, k):
    num_nodes = adj_matrix.shape[0]
    assignment = np.zeros(num_nodes, dtype=int)
    improved = True
    iteration_count = 0
    max_iterations = 100
    while improved and iteration_count < max_iterations:
        improved = False
        new_assignment = np.copy(assignment)
        for node in range(num_nodes):
            best_color = least_conflicting_color(adj_matrix, assignment, num_nodes, node, k)
            if best_color != assignment[node]:
                assignment[node] = best_color
                # new_assignment[node] = best_color
                improved = True
        # assignment = new_assignment
        iteration_count += 1

    # return [a+1 for a in assignment]
    return assignment









import cvxpy as cp
import numpy as np

def max_3_cut_sdp(adj_matrix, k=3, dim=None):
    n = len(adj_matrix)  # Number of vertices
    dim = dim or n

    # Define the SDP variables
    X = cp.Variable((n, dim), symmetric=True)

    # Define the constraints
    constraints = [X >> 0]  # X is positive semidefinite
    for i in range(n):
        constraints.append(X[i, i] == 1)  # Unit length constraint

    for u in range(n):
        for v in range(n):
            if adj_matrix[u, v] == 1:
                constraints.append(X[u, v] >= -1/(k-1))  # Distance constraint for adjacent nodes

    # Define the objective function
    objective = cp.Maximize(cp.sum(cp.multiply(adj_matrix, 1 - X) * (k-1)/k))

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    eigenvalues, eigenvectors = np.linalg.eigh(X.value)
    eigenvalues[eigenvalues < 0] = 0
    vectors = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    reconstructed_X = vectors @ vectors.T
    return vectors

def max_3_cut_sdp2(adj_matrix, k=3, dim=64):
    n = len(adj_matrix)  # Number of vertices
    dim = dim or n  # Dimension of the vectors, can be adjusted as needed

    # Define the variables
    V = cp.Variable((n, dim))

    # Objective function
    objective = cp.Maximize(cp.sum([(1 - cp.sum(cp.multiply(V[i], V[j]))) * (k-1)/k for i in range(n) for j in range(i+1, n) if adj_matrix[i, j] > 0]))

    # Constraints
    constraints = []
    # constraints += [cp.norm(V[i]) == 1 for i in range(n)]  # v_i \cdot v_i = 1
    # constraints += [cp.sum(cp.multiply(V[i], V[j])) >= -1/(k-1) for i in range(n) for j in range(n) if adj_matrix[i, j] > 0]  # v_i \cdot v_j >= -1/(k-1)

    # Define the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve()

    return V.value

def generate_random_unit_vector(n):
    vec = np.random.normal(0, 1, n)
    return vec / np.linalg.norm(vec)

def assign_clusters(V, k):
    n, d = V.shape
    random_vectors = np.random.randn(k, d)  # Generate random vectors from a normal distribution

    # Normalize the random vectors
    random_vectors = random_vectors / np.linalg.norm(random_vectors, axis=1, keepdims=True)

    # Compute dot products
    dot_products = V @ random_vectors.T

    # Assign clusters based on maximum dot product
    clusters = np.argmax(dot_products, axis=1)

    return clusters

def solve_max_3_cut_frieze_jerrum(adj_matrix, k=3, dim=None):
    V = max_3_cut_sdp(adj_matrix, k=k, dim=dim)
    clusters = assign_clusters(V, k)
    return V, clusters


























import networkx as nx
import picos
from operator import itemgetter
import numpy as np
import cvxopt as cvx
import cvxopt.lapack
from cvxopt.base import matrix as cvx_matrix
from scipy.linalg import sqrtm
from scipy.linalg import eigh
from scipy.linalg import eig
from picos import RealVariable, BinaryVariable


def solve_max_k_cut_sdp(G: nx.Graph, k: int, weight: str = "weight"):
    """
    solve the sdp problem with object of max-3-cut and G as an input.
    Frieze & Jerrum: https://www.math.cmu.edu/~af1p/Texfiles/cuts.pdf
    :param G: undirected graph
    :return: embedding
    """

    num_nodes = G.number_of_nodes()
    if num_nodes <= 1:
        return None

    sum_edges_weight = sum(map(itemgetter(weight), map(itemgetter(2), G.edges(data=True))))
    if sum_edges_weight == 0:
        return None

    maxcut = picos.Problem()
    # print(f'num nodes: {num_nodes}')
    # Add the symmetric matrix variable.
    X = maxcut.add_variable('X', (num_nodes, num_nodes), 'symmetric')

    # Retrieve the Laplacian of the graph.
    LL = ((k - 1) / (2 * k)) * nx.laplacian_matrix(G, weight=weight).todense()
    L = picos.new_param('L', LL)

    # Constrain X to have ones on the diagonal.
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                maxcut.add_constraint(X[i, j] >= (-1 / (k - 1)))
    # maxcut.add_constraint(X >= (-1 / (k - 1)))

    maxcut.add_constraint(picos.diag_vect(X) == 1)

    # Constrain X to be positive semidefinite.
    maxcut.add_constraint(X >> 0)

    # Set the objective.
    maxcut.set_objective('max', L | X)

    # Solve the problem.
    maxcut.solve(verbose=0, solver='cvxopt')
    indexed_nodes = list(G.nodes)

    ### Perform the random relaxation
    # Use a fixed RNG seed so the result is reproducable.
    # cvx.setseed(1919)

    # Perform a Cholesky factorization (in the lower triangular part of the matrix)
    # https://en.wikipedia.org/wiki/Cholesky_decomposition#Proof_for_positive_semi-definite_matrices
    # https://en.wikipedia.org/wiki/Square_root_of_a_matrix
    # https://math.stackexchange.com/questions/1801403/decomposition-of-a-positive-semidefinite-matrix
    # https://en.wikipedia.org/wiki/Matrix_decomposition#Cholesky_decomposition
    # https://stackoverflow.com/questions/5563743/check-for-positive-definiteness-or-positive-semidefiniteness
    # https://en.wikipedia.org/wiki/Matrix_decomposition#Cholesky_decomposition:~:text=Cholesky%20decomposition%5Bedit%5D
    # https://proceedings.neurips.cc/paper/2021/file/45c166d697d65080d54501403b433256-Paper.pdf
    D, V = eigh(X.value)
    # Z = (V * np.sqrt(D)) @ V.T
    s = np.diag(np.abs(D))
    z = np.sqrt(s)
    # V = np.linalg.cholesky(X.value)
    V = V @ z
    return maxcut.value, {indexed_nodes[i]: np.expand_dims(np.array(V[i, :]), axis=0) for i in range(num_nodes)}


def generate_random_unit_vector(dim, num_of_vectors):
    random_unit_vectors = []
    for i in range(num_of_vectors):
        v = np.random.rand(dim)
        random_unit_vectors.append(v / (v ** 2).sum() ** 0.5)
    return random_unit_vectors


def classify_nodes(G: nx.Graph, nodes_embeddings, k):
    unit_vectors = generate_random_unit_vector(G.number_of_nodes(), k)
    result = {}
    indexed_nodes = list(G.nodes)
    for node in indexed_nodes:
        classify = 0
        min_distance = 1000000
        for i, vector in enumerate(unit_vectors):
            distance = np.linalg.norm(nodes_embeddings[node] - vector)
            if distance < min_distance:
                classify = i
                min_distance = distance
        result[node] = classify
    return result


def run_max_k_cut(G: nx.Graph, k):
    value, node_id_to_embedding = solve_max_k_cut_sdp(G, k)
    node_id_to_classification = classify_nodes(G, node_id_to_embedding, k)
    return value, node_id_to_embedding, node_id_to_classification

def has_cycle(graph):
    # Iterative DFS to detect cycles
    # def iterative_dfs(start_node):
    #     stack = [(start_node, -1)]
    #     visited = [False] * len(graph)
    #     rec_stack = [False] * len(graph)
    #     while stack:
    #         node, parent = stack.pop()
    #         if not visited[node]:
    #             visited[node] = True
    #             rec_stack[node] = True
    #             for neighbor in range(len(graph)):
    #                 if graph[node][neighbor] == 1:
    #                     if not visited[neighbor]:
    #                         stack.append((neighbor, node))
    #                     elif rec_stack[neighbor]:
    #                         return True
    #             rec_stack[node] = False
    #     return False
    def dfs(adj):
        stack = [0]
        visited = [False] * len(adj)
        ccs = []
        cycle = False
        while sum(visited) != len(visited):
            cc = 0
            while stack:
                node = stack.pop()
                # print(node)
                if visited[node]:
                    cycle = True
                    continue
                cc += 1
                visited[node] = True
                stack.extend([n for n, v in enumerate(adj[node]) if v == 1])
            if False in visited: stack.append(visited.index(False))
            ccs.append(cc)
        return cycle, ccs

    # for node in range(len(graph)):
    #     if iterative_dfs(node):
    #         return True
    return dfs(graph)
    # return False

def find_largest_degree_node(graph):
    degrees = np.sum(graph, axis=0) + np.sum(graph, axis=1)
    return np.argmax(degrees)

def remove_cycles(graph):
    # Initialize an empty list to keep track of removed nodes
    removed_node_list = []

    while has_cycle(graph)[0]:
        node = find_largest_degree_node(graph)
        graph[node, :] = 0
        graph[:, node] = 0
        removed_node_list.append(node)

    return removed_node_list

def find_least_common_neighbor_color(adj, assignment, feautres, k=3):
    n = adj.shape[0]
    color_counts = np.zeros((n, k), dtype=int)  # To store counts of neighbors for each color
    conflict_sums = np.zeros(n, dtype=int)  # To store the sum of conflicts for other colors
    neighbor_dist = np.zeros(n)
    degrees = adj.sum(axis=1)
    conflict_avg = np.zeros(n)
    conflict_max = np.zeros(n)
    conflict_min = np.zeros(n)

    for i in range(n):
        neighbors = torch.where(adj[i] == 1)[0]
        for neighbor in neighbors:
            color_counts[i][assignment[neighbor] - 1] += 1  # -1 to convert to 0-based index
            neighbor_dist[i] += np.linalg.norm(feautres[i] - feautres[neighbor])
        if len(neighbors)>0:neighbor_dist[i] = neighbor_dist[i] / len(neighbors)

    least_common_colors = np.zeros(n, dtype=int)
    for i in range(n):
        current_color_index = assignment[i] - 1  # Convert to 0-based index
        min_count = n+100 #color_counts[i][current_color_index]
        least_common_color_index = current_color_index

        for j in range(k):
            if j==current_color_index:
                continue
            if color_counts[i][j] < min_count or (color_counts[i][j] == min_count and j == current_color_index):
                min_count = color_counts[i][j]
                least_common_color_index = j

        least_common_colors[i] = least_common_color_index + 1  # Convert back to 1-based index
        conflict_sums[i] = np.sum(color_counts[i]) - color_counts[i][least_common_color_index]
        conflict_avg[i] = conflict_sums[i]/degrees[i] if degrees[i] > 0 else 0
        conflict_max[i] = max(color_counts[i][[j for j in range(k) if j != current_color_index]])
        conflict_min[i] = min(color_counts[i][[j for j in range(k) if j != current_color_index]])

    return least_common_colors, [conflict_sums, conflict_avg, conflict_max, conflict_min], neighbor_dist




import concurrent.futures

def compare_pairs(assignments, start, end):
    last_same = {}
    last_different = {}
    n = len(assignments[0])
    for i in range(start, end):
        for j in range(i + 1, n):
            last_same[(i, j)] = (0, [])
            last_different[(i, j)] = (0, [])
            for a_i, assignment in enumerate(assignments):
                if assignment[i] == assignment[j]:
                    count_same, list_same = last_same[(i, j)]
                    last_same[(i, j)] = (count_same + 1, list_same + [a_i])
                else:
                    count_different, list_different = last_different[(i, j)]
                    last_different[(i, j)] = (count_different + 1, list_different + [a_i])

    return last_same, last_different

def last_change_compare(assignments):
    n = len(assignments[0])
    num_threads = 20  # Adjust the number of threads as needed
    chunk_size = n // num_threads

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(0, n, chunk_size):
            start = i
            end = min(i + chunk_size, n)
            futures.append(executor.submit(compare_pairs, assignments, start, end))

    last_same = {}
    last_different = {}
    for future in concurrent.futures.as_completed(futures):
        same, different = future.result()
        last_same.update(same)
        last_different.update(different)

    return last_same, last_different

# def last_change_compare(assignments):
#     last_same = {}
#     last_different = {}
#     n = len(assignments[0])
#
#     for i in range(n):
#         for j in range(i+1, n):
#             last_same[(i,j)] = -1
#             last_different[(i,j)] = -1
#             for a_i, assignment in enumerate(assignments):
#                 if assignment[i] == assignment[j]:
#                     last_same[(i,j)] = a_i
#                 else:
#                     last_different[(i,j)] = a_i
#     return last_same, last_different

def calc_dist_in_iteratoin(adj, features):
    n = adj.shape[0]
    dist = np.zeros(n)
    for i in range(n):
        neighbors = torch.where(adj[i] == 1)[0]
        dist[i] = np.mean(np.linalg.norm(features[i] - features[neighbors]))
    return dist
def calc_dist_over_iteartions(adj, features_iteartions):
    dist = []
    for features in features_iteartions:
        dist.append(calc_dist_in_iteratoin(adj, features))
    return dist

import numpy as np
import plotly.express as px
import pandas as pd
from scipy.stats import spearmanr as sci_spearmanr


def plot_conf_over_time(distances_over_time, closesed_index):
    distances_over_time_range = distances_over_time[:closesed_index + 1]
    # Sort track_i by np.array(distances_over_time[-1]).argsort()
    sorted_indices = np.array(distances_over_time_range[-1]).argsort()
    sorted_distances = np.array(distances_over_time_range)[:, sorted_indices]
    df = pd.DataFrame(sorted_distances, columns=[f"track_{i}" for i in sorted_indices])

    spear = -2
    spear_p = -2
    if len(sorted_distances[1:-1]) >0:
        mean_over_time = sorted_distances[1:-1].mean(0)
        s = sci_spearmanr(mean_over_time, sorted_distances[-1])
        spear = s.statistic
        spear_p = s.pvalue

    df['time'] = list(range(len(distances_over_time_range)))
    df_melted = df.melt(id_vars='time', var_name='track', value_name='distance')

    # Create the animated line plot
    fig = px.line(df_melted, x='time', y='distance', animation_frame='track')
    # fig.show()
    return spear, spear_p

def confidance_same_pair(adj, local_assignments, distances_over_time):
    n = adj.shape[0]
    percentile_range = [10, 30, 50, 70, 90]
    nieghboring_nodes = [(i,j) for i in range(n) for j in range(i+1, n) if adj[i,j]==1]
    # histogram = [[0]*11 for _ in range(len(distances_over_time))]

    same_historgrams = [[0]*(len(percentile_range)+1) for _ in range(len(distances_over_time))]
    counts_histograms = [[0]*(len(percentile_range)+1) for _ in range(len(distances_over_time))]

    for iteration, (local_assignment, distance_over_time) in enumerate(zip(local_assignments, distances_over_time)):
        current_distances = {pair: (distance_over_time[pair[0]]+distance_over_time[pair[1]])/2 for pair in nieghboring_nodes }
        sorted = np.array(list(current_distances.values()))
        sorted.sort()
        per = [np.percentile(sorted, i/100) for i in percentile_range]
        for pair in current_distances:
            index = np.searchsorted(per, current_distances[pair], side='right')
            counts_histograms[iteration][index] += 1
            if local_assignment[pair[0]] == local_assignment[pair[1]]:
                same_historgrams[iteration][index] += 1
    same_historgrams = np.array(same_historgrams)
    counts_histograms = np.array(counts_histograms)
    counts_histograms[counts_histograms == 0] = -1
    result = np.where(counts_histograms != -1, same_historgrams / counts_histograms, -1)
    return numpy.array([sum(row[row!=-1])/len(row[row!=-1]) if len(row[row!=-1])>0 else -1 for row in result.T])




def confidance_pair(adj, local_assignments, distances_over_time, closesed_index, supports):
    local_assignments = local_assignments[:closesed_index+1]
    distances_over_time = distances_over_time[:closesed_index+1]
    n = adj.shape[0]
    percentile_range = [10, 20, 30, 50, 70, 80, 90]


    below_pairs = {(percentile_range[p1],percentile_range[p2]):0 for p1 in range(len(percentile_range)) for p2 in range(p1, len(percentile_range))}
    above_pairs = {(percentile_range[p1],percentile_range[p2]):0 for p1 in range(len(percentile_range)) for p2 in range(p1, len(percentile_range))}

    below_pairs_count = {(percentile_range[p1],percentile_range[p2]):0 for p1 in range(len(percentile_range)) for p2 in range(p1, len(percentile_range))}
    above_pairs_count = {(percentile_range[p1],percentile_range[p2]):0 for p1 in range(len(percentile_range)) for p2 in range(p1, len(percentile_range))}


    nieghboring_nodes = [(i,j) for i in range(n) for j in range(i+1, n) if adj[i,j]==1]
    # histogram = [[0]*11 for _ in range(len(distances_over_time))]


    distance_over_time = distances_over_time[-1]
    support_over_time = supports[-1][-1]
    distance_percentile_70 = np.percentile(distance_over_time, 70)
    support_percentile_70 = np.percentile(support_over_time, 70)
    distance_nodes_above_70 = set(np.where(distance_over_time >= distance_percentile_70)[0])
    support_nodes_above_70 = set(np.where(support_over_time >= support_percentile_70)[0])
    distance_percentile_80 = np.percentile(distance_over_time, 80)
    support_percentile_80 = np.percentile(support_over_time, 80)
    distance_nodes_above_80 = set(np.where(distance_over_time >= distance_percentile_80)[0])
    support_nodes_above_80 = set(np.where(support_over_time >= support_percentile_80)[0])
    distance_percentile_90 = np.percentile(distance_over_time, 90)
    support_percentile_90 = np.percentile(support_over_time, 90)
    distance_nodes_above_90 = set(np.where(distance_over_time >= distance_percentile_90)[0])
    support_nodes_above_90 = set(np.where(support_over_time >= support_percentile_90)[0])

    nodes_above_70 = distance_nodes_above_70.intersection(support_nodes_above_70)
    nodes_above_80 = distance_nodes_above_80.intersection(support_nodes_above_80)
    nodes_above_90 = distance_nodes_above_90.intersection(support_nodes_above_90)


    support_percentile_of_70 = []
    support_percentile_of_80 = []
    support_percentile_of_90 = []
    interstion_70 = []
    interstion_80 = []
    interstion_90 = []

    for iteration, (local_assignment, distance_over_time) in enumerate(zip(local_assignments, distances_over_time)):
        if iteration <= 0.2*closesed_index: continue
        support = supports[iteration][-1]
        # support = [abs(supports[iteration][1][i]-s/2)/s if s>0 else 1 for i,s in enumerate(support)]
        # support = [s[-1] for i,s in enumerate(support)]


        distance_current_percentile_70 = np.percentile(distance_over_time, 70)
        support_current_percentile_70 = np.percentile(support, 70)
        distance_current_nodes_above_70 = np.where(distance_over_time >= distance_current_percentile_70)[0]
        support_current_nodes_above_70 = np.where(support >= support_current_percentile_70)[0]
        distance_current_percentile_80 = np.percentile(distance_over_time, 80)
        support_current_percentile_80 = np.percentile(support, 80)
        distance_current_nodes_above_80 = np.where(distance_over_time >= distance_current_percentile_80)[0]
        support_current_nodes_above_80 = np.where(support >= support_current_percentile_80)[0]
        distance_current_percentile_90 = np.percentile(distance_over_time, 90)
        support_current_percentile_90 = np.percentile(support, 90)
        distance_current_nodes_above_90 = np.where(distance_over_time >= distance_current_percentile_90)[0]
        support_current_nodes_above_90 = np.where(support >= support_current_percentile_90)[0]

        current_nodes_above_70 = set(distance_current_nodes_above_70).intersection(set(support_current_nodes_above_70))
        current_nodes_above_80 = set(distance_current_nodes_above_80).intersection(set(support_current_nodes_above_80))
        current_nodes_above_90 = set(distance_current_nodes_above_90).intersection(set(support_current_nodes_above_90))

        for node in current_nodes_above_70:
            # support_percentile_of_70.append(percentileofscore(support, support[node]))
            support_percentile_of_70.append(percentileofscore(distance_over_time, distance_over_time[node]))
        for node in current_nodes_above_80:
            # support_percentile_of_80.append(percentileofscore(support, support[node]))
            support_percentile_of_80.append(percentileofscore(distance_over_time, distance_over_time[node]))
        for node in current_nodes_above_90:
            # support_percentile_of_90.append(percentileofscore(support, support[node]))
            support_percentile_of_90.append(percentileofscore(distance_over_time, distance_over_time[node]))

        if iteration < closesed_index:
            if len(nodes_above_80) > 0:
                interstion_80.append(
                    len(nodes_above_80.intersection(set(current_nodes_above_80)))/
                    len(nodes_above_80)
                )
            if len(nodes_above_70) > 0:
                interstion_70.append(
                    len(nodes_above_70.intersection(set(current_nodes_above_70)))/
                    len(nodes_above_70)
                )

            if len(nodes_above_90) > 0:
                interstion_90.append(
                    len(nodes_above_90.intersection(set(current_nodes_above_90)))/
                    len(nodes_above_90)
                )

        current_percentiles = np.array([np.percentile(distance_over_time, p) for p in percentile_range])
        current_support_percentiles = np.array([np.percentile(support, p) for p in percentile_range])
        node_percentile = {}
        for u,v in nieghboring_nodes:
            # du = percentileofscore(distance_over_time, distance_over_time[u])
            du = distance_over_time[u]
            # dv = percentileofscore(distance_over_time, distance_over_time[v])
            dv = distance_over_time[v]
            # su = percentileofscore(support, support[u])
            su = support[u]
            # sv = percentileofscore(support, support[v])
            sv = support[v]

            for p in range(len(percentile_range)):
                if du <= current_percentiles[p] and su <= current_support_percentiles[p] and dv <= current_percentiles[p] and sv <= current_support_percentiles[p]:
                    below_pairs_count[(percentile_range[p],percentile_range[p])] += 1
                    if local_assignment[u] == local_assignment[v]:
                        below_pairs[(percentile_range[p],percentile_range[p])] += 1
                if du >= current_percentiles[p] and su<= current_support_percentiles[p] and dv >= current_percentiles[p] and sv <= current_support_percentiles[p]:
                    above_pairs_count[(percentile_range[p], percentile_range[p])] += 1
                    if local_assignment[u] == local_assignment[v]:
                        above_pairs[(percentile_range[p],percentile_range[p])] += 1

            # for p1 in range(len(percentile_range)):
            #     for p2 in range(p1, len(percentile_range)):
            #         if du <= current_percentiles[p1] and dv <= current_percentiles[p2]:
            #             below_pairs_count[(percentile_range[p1],percentile_range[p2])] += 1
            #             if local_assignment[u] == local_assignment[v]:
            #                 below_pairs[(percentile_range[p1],percentile_range[p2])] += 1
            #         if du >= p1 and dv >= p2:
            #             above_pairs_count[(percentile_range[p1], percentile_range[p2])] += 1
            #             if local_assignment[u] == local_assignment[v]:
            #                 above_pairs[(percentile_range[p1],percentile_range[p2])] += 1

    above= {
        str(k): v/above_pairs_count[k] if above_pairs_count[k] > 0 else -1 for k,v in above_pairs.items()
    }
    below={
        str(k): v/below_pairs_count[k] if below_pairs_count[k] > 0 else -1 for k,v in below_pairs.items()
    }
    interstion_70 = np.mean(interstion_70).item() if len(interstion_70)>0 else -1
    interstion_80 = np.mean(interstion_80).item() if len(interstion_80)>0 else -1
    interstion_90 = np.mean(interstion_90).item() if len(interstion_90)>0 else -1
    support_percentile_of_70 = np.array(support_percentile_of_70)
    support_percentile_of_80 = np.array(support_percentile_of_80)
    support_percentile_of_90 = np.array(support_percentile_of_90)
    # print("intersections", interstion_70, interstion_80, interstion_90, np.mean(support_percentile_of_70[support_percentile_of_70>=0]), np.mean(support_percentile_of_80[support_percentile_of_80>=0]), np.mean([support_percentile_of_90>=0]), len(support_percentile_of_90>=0))
    return above, below, interstion_70, interstion_80 , interstion_90, support_percentile_of_70, support_percentile_of_80, support_percentile_of_90


def max_clique_sdp():
    import cvxpy as cp
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import plotly.express as px

    # Function to generate a planted clique graph G(n, p, k) as a numpy adjacency matrix
    def generate_planted_clique(n, p, k):
        # Generate a random adjacency matrix for G(n, p)
        A = (np.random.rand(n, n) < p).astype(float)
        np.fill_diagonal(A, 0)  # No self-loops

        # Add a planted clique of size k
        clique_nodes = np.random.choice(n, k, replace=False)
        for i in clique_nodes:
            for j in clique_nodes:
                if i != j:
                    A[i, j] = 1
                    A[j, i] = 1  # Ensure symmetry

        return A, clique_nodes

    # Function to solve the SDP relaxation of the max clique problem
    def solve_sdp_max_clique(A):
        n = A.shape[0]

        # Define the SDP problem
        X = cp.Variable((n, n), symmetric=True)

        # Objective: Maximize the trace of X (trace maximization helps find the largest submatrix)
        # objective = cp.Maximize(cp.trace(X))
        objective = cp.Maximize(cp.sum(X))

        # Constraints
        constraints = [
                          X >> 0,  # X must be positive semi-definite
                          cp.trace(X) == 1  # Diagonal elements must be 1 (each node has to correlate with itself)
                      ] + [X[i, j] == 0 for i in range(n) for j in range(n) if A[i, j] == 0 and i != j
                           # Non-edges have zero correlation
                           ]

        # Solve the SDP
        prob = cp.Problem(objective, constraints)
        prob.solve()
        X_value = X.value
        print(X_value)
        eigvals, eigvecs = np.linalg.eigh(X_value)

        # Find the largest eigenvalue and corresponding eigenvector
        idx = np.argmax(eigvals)
        v = eigvecs[:, idx]

        # Recover the clique by selecting the nodes with large projections
        threshold = 1e-3
        clique_nodes = np.where(v > threshold)[0]

        # Return the solution matrix X
        return X_value, len(clique_nodes), clique_nodes, v

    # PCA and Plotting
    def plot_sdp_solution(X, planted, found):
        # Apply PCA to the SDP solution matrix
        pca = PCA(n_components=2)
        pos = pca.fit_transform(X)

        # Plot using Plotly
        fig = px.scatter(x=pos[:, 0], y=pos[:, 1], title='2D Embedding of SDP Solution color planted cliques',
                        color=["0" if i in planted else "1" for i in range(n)],
                         labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
                         size_max=10)

        # Show the plot
        fig.show()
        fig = px.scatter(x=pos[:, 0], y=pos[:, 1], title='2D Embedding of SDP Solution color found cliques',
                         color=["0" if i in found else "1" for i in range(n)],
                         labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
                         size_max=10)

        # Show the plot
        fig.show()

    n = 50  # Number of nodes
    p = 0.3  # Probability for edge creation
    k = 10  # Size of the planted clique

    # Generate a planted clique graph as an adjacency matrix
    A, planted_clique = generate_planted_clique(n, p, k)

    # Solve for maximum clique using SDP
    X, clique_size, clique_nodes, v = solve_sdp_max_clique(A)

    print(f"SDP Estimated Clique Size: {clique_size}")
    print(f"SDP Estimated Clique Nodes: {clique_nodes}, {planted_clique}")
    print(f"number of non zeros: {sum(X.sum(0) != 0)}")
    print("v", v)

    # Visualize the graph in 2D PCA space
    plot_sdp_solution(X, planted_clique, clique_nodes)

