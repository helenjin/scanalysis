# palantir without Multibranch or scdata objects

import numpy as np
import pandas as pd
import random

import networkx as nx
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr, entropy
from numpy.linalg import inv

from copy import deepcopy

from sklearn.cluster import SpectralClustering, KMeans
from scipy.sparse import csr_matrix, find
from scipy.stats import entropy

import time 
from scipy.stats import norm  
from scipy.cluster import hierarchy


# Max min sampling of waypoints
def max_min_sampling(data, num_waypoints, knn, n_jobs, flock):
    
    waypoint_set, waypoint_sampling_dim = _max_min_worker(data, num_waypoints)
    
    # Update
    waypoints = waypoint_set
    waypoints_dim = pd.Series(waypoint_sampling_dim, index = waypoints)
    

    # Flock waypoints if specified
  #  waypoints, waypoints_dim = flock_waypoints(data, waypoints, waypoints_dim, knn, n_jobs, flock )
    
    return waypoints, waypoints_dim


def _max_min_worker(data, num_waypoints):
    # Max min  sampling of landmarks
    waypoint_set = list()
    no_iterations = int((num_waypoints)/data.shape[1])
    waypoint_sampling_dim = np.array([], dtype=np.int64)

    # Sample for all diffusion components
    N = data.shape[0]
    for ind in data.columns:
        # Data vector
        vec = np.ravel(data[ind])

        # Random initialzlation
        iter_set = random.sample(range(N), 1)

        # Distances along the component
        dists = np.zeros( [N, no_iterations] )
        dists[:, 0] = abs(vec - data[ind].values[iter_set])
        for k in range(1, no_iterations):
            # Minimum distances across the current set
            min_dists = dists[:, 0:k].min(axis = 1)

            # Point with the maximum of the minimum distances is the new waypoint
            new_wp = np.where(min_dists == min_dists.max())[0][0]
            iter_set.append( new_wp )

            # Update distances
            dists[:, k] = abs(vec - data[ind].values[new_wp])

        # Update global set
        waypoint_set = waypoint_set +  iter_set
        waypoint_sampling_dim = np.append( waypoint_sampling_dim, np.repeat( ind, no_iterations )) 

    # Set waypoints to the standard variable
    waypoint_set = data.index[waypoint_set]
    # Remove duplicates
    waypoint_sampling_dim = waypoint_sampling_dim[~waypoint_set.duplicated()]
    waypoint_set = waypoint_set[~waypoint_set.duplicated()]
    waypoint_sampling_dim = pd.Series(waypoint_sampling_dim, index=waypoint_set)

    return waypoint_set, waypoint_sampling_dim

# Function to determine weights
def _weighting_scheme(D, voting_scheme):
    if voting_scheme == 'uniform':
        W = pd.DataFrame(np.ones(D.shape), 
                index = D.index, columns=D.columns)
    elif voting_scheme == 'exponential':
        # sdv = D.std().mean() * 3
        # sdv = np.std( np.ravel(D) )
        # Silverman's approximation for standard deviation
        sdv = np.std(np.ravel(D)) * 1.06 * len(np.ravel(D)) **(-1/5)
        W = np.exp( -.5 * np.power((D / sdv), 2))

        # W = np.exp(-D/(np.std(np.ravel(D))))

    elif voting_scheme == 'linear':
        W = np.matlib.repmat(D.max(), len(D), 1) - D

    # Stochastize
    W = W / W.sum()
    return W




def _stationary_distribution(data, waypoints, knn, n_jobs, trajectory):
    # kNN graph 
    n_neighbors = knn
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean',
        n_jobs=n_jobs).fit(data.loc[waypoints,:])
    # nbrs = NearestNeighbors(n_neighbors=self.knn, metric='cosine', algorithm='brute',
    #     n_jobs=1).fit(self.data.loc[self.waypoints,:])
    kNN = nbrs.kneighbors_graph(data.loc[waypoints,:], mode='distance' ) 
    dist,ind = nbrs.kneighbors(data.loc[waypoints,:])

    # Standard deviation allowing for "back" edges
    adaptive_std = np.ravel(dist[:, int(np.floor(n_neighbors / 3)) - 1])

    # Directed graph construction
    # Trajectory position of all the neighbors
    traj_nbrs = pd.DataFrame(trajectory[np.ravel(waypoints[ind])].values.reshape( 
        [len(waypoints), n_neighbors]), index=waypoints)

    # Remove edges that move backwards in trajectory except for edges that are within 
    # the computed standard deviation
    rem_edges = traj_nbrs.apply(lambda x : x < trajectory[traj_nbrs.index] - adaptive_std )
    rem_edges = rem_edges.stack()[rem_edges.stack()]

    # Determine the indices and update adjacency matrix
    cell_mapping = pd.Series(range(len(waypoints)), index=waypoints)
    x = list(cell_mapping[rem_edges.index.get_level_values(0)])
    y = list(rem_edges.index.get_level_values(1)) 
    # Update adjacecy matrix
    kNN[x, ind[x, y]] = 0  

    # Affinity matrix and markov chain
    x, y, z = find(kNN)
    aff = np.exp(-(z ** 2)/(adaptive_std[x] ** 2)  * 0.5 \
         -(z ** 2)/(adaptive_std[y] ** 2)  * 0.5 )
    W = csr_matrix((aff, (x, y)), [len(waypoints), len(waypoints)])
    # W = csr_matrix((1-z, (x, y)), [len(self.waypoints), len(self.waypoints)])

    # Probabilities
    D = np.ravel(W.sum(axis = 1))
    x, y, z = find(W)
    T = csr_matrix(( z / D[x], (x, y)), [len(waypoints), len(waypoints)])

    from scipy.sparse.linalg import eigs
    vals, vecs = eigs(T.T, 10)

    ranks = np.abs(np.real(vecs[:, np.argsort(vals)[-1]]))
    ranks = pd.Series(ranks, index=waypoints)

    # Cutoff and intersection with the boundary cells
    cutoff = norm.ppf(0.9999, loc=np.median(ranks), 
        scale=np.median(np.abs((ranks - np.median(ranks)))))
    dm_boundaries = pd.Index(set(data.idxmax()).union(data.idxmin()))
    cells = (ranks.index[ ranks > cutoff].intersection( dm_boundaries ))

    # Clusters cells
    Z = hierarchy.linkage(pairwise_distances(data.loc[cells,:]))
    clusters = pd.Series(hierarchy.fcluster(Z, 1, 'distance'), index=cells)  

    # IDentify cells with maximum trajectory for each cluster
    cells = trajectory[clusters.index].groupby(clusters).idxmax()

    return cells, ranks



def _differentiation_entropy(data, waypoints, knn, n_jobs, start_cell, trajectory, boundary_cells=None):

    # Identify boundary cells
    if boundary_cells is None:
        print('Identifying end points...')
        res = _determine_end_points(data, waypoints, knn, n_jobs, trajectory, start_cell)
        boundary_cells = res[0]

    # Markov chain construction
    print('Markov chain construction...')

    # kNN graph 
    n_neighbors = knn
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean',
        n_jobs=n_jobs).fit(data.loc[waypoints,:])
    # nbrs = NearestNeighbors(n_neighbors=self.knn, metric='cosine', algorithm='brute',
    #     n_jobs=1).fit(self.data.loc[self.waypoints,:])
    kNN = nbrs.kneighbors_graph(data.loc[waypoints,:], mode='distance' ) 
    dist,ind = nbrs.kneighbors(data.loc[waypoints,:])

    # Standard deviation allowing for "back" edges
    adaptive_std = np.ravel(dist[:, int(np.floor(n_neighbors / 3)) - 1])

    # Directed graph construction
    # Trajectory position of all the neighbors
    traj_nbrs = pd.DataFrame(trajectory[np.ravel(waypoints[ind])].values.reshape( 
        [len(waypoints), n_neighbors]), index=waypoints)

    # Remove edges that move backwards in trajectory except for edges that are within 
    # the computed standard deviation
    rem_edges = traj_nbrs.apply(lambda x : x < trajectory[traj_nbrs.index] - adaptive_std )
    rem_edges = rem_edges.stack()[rem_edges.stack()]

    # Determine the indices and update adjacency matrix
    cell_mapping = pd.Series(range(len(waypoints)), index=waypoints)
    x = list(cell_mapping[rem_edges.index.get_level_values(0)])
    y = list(rem_edges.index.get_level_values(1)) 
    # Update adjacecy matrix
    kNN[x, ind[x, y]] = 0  

    # Affinity matrix and markov chain
    x, y, z = find(kNN)
    aff = np.exp(-(z ** 2)/(adaptive_std[x] ** 2)  * 0.5 \
         -(z ** 2)/(adaptive_std[y] ** 2)  * 0.5 )
    W = csr_matrix((aff, (x, y)), [len(waypoints), len(waypoints)])
    # W = csr_matrix((1-z, (x, y)), [len(self.waypoints), len(self.waypoints)])

    # Probabilities
    D = np.ravel(W.sum(axis = 1))
    x, y, z = find(W)
    T = csr_matrix(( z / D[x], (x, y)), [len(waypoints), len(waypoints)])


    # Absorption states should not have outgoing edges
    abs_states = np.where(waypoints.isin(boundary_cells))[0]
    # Reset absorption state affinities by Removing neigbors
    T[abs_states,:] = 0 
    # Diagnoals as 1s
    T[abs_states, abs_states] = 1



    # Fundamental matrix and absorption probabilities
    print('Computing fundamental matrix and absorption probabilities...')
    # Transition states
    trans_states = list(set(range(len(waypoints))).difference(abs_states))

    # Q matrix
    Q = T[trans_states,:][:, trans_states]
    # Fundamental matrix
    mat = np.eye(Q.shape[0]) - Q.todense()
    N = inv(mat)

    # Absorption probabilities
    abs_probabilities = np.dot(N, T[trans_states,:][:, abs_states].todense())
    abs_probabilities = pd.DataFrame(abs_probabilities, 
        index=waypoints[trans_states],
        columns=waypoints[abs_states])
    abs_probabilities[abs_probabilities < 0] = 0

    # Entropy
    ent = abs_probabilities.apply(entropy, axis=1)

    return ent, abs_probabilities


def _shortest_path_helper(cell, adj):
    # NOTE: Graph construction is parallelized since constructing the graph outside was creating lock issues
    graph = nx.Graph(adj)
    return pd.Series(nx.single_source_dijkstra_path_length(graph, cell))

def run_multibranch(data_, DMEigs, DMEigVals, dm_eigs, start_cell, num_waypoints, knn=25, flock=2, n_jobs=1, voting_scheme='exponential', max_iterations=25):
    # # ################################################
    
    # Multi scale distance
    eig_vals = np.ravel(DMEigVals.values[dm_eigs])
    data_  = pd.DataFrame.transpose(data_)
    data = DMEigs.values[:, dm_eigs] * (eig_vals / (1-eig_vals))
    data = pd.DataFrame( data, index=data_.columns, columns=dm_eigs )
    
    # # Sample waypoints
    print('Sampling and flocking waypoints...')
    start =time.time()

    # Append start cell
    if isinstance(num_waypoints, int):
        waypoints, waypoints_dim = max_min_sampling( data, num_waypoints, knn, n_jobs, flock )
        waypoints = pd.Index(waypoints.difference([start_cell]).unique())
    else:
        waypoints = num_waypoints
    waypoints = pd.Index([start_cell]).append(waypoints)
    end =time.time()
    print('Time for determining waypoints: {} minutes'.format((end - start)/60))


    # ################################################
    # Shortest path distances to determine trajectories
    print('Shortest path distances...')
    start =time.time()
    nbrs = NearestNeighbors(n_neighbors=knn, 
            metric='euclidean', n_jobs=n_jobs).fit(data) 
    adj = nbrs.kneighbors_graph(data, mode='distance')

    # Distances
    dists = Parallel(n_jobs=n_jobs)(
        delayed(_shortest_path_helper)(np.where(data.index == cell)[0][0], adj) 
            for cell in waypoints)

    # Convert to distance matrix
    D = pd.DataFrame(0.0, index=waypoints, columns=data.index)
    for i, cell in enumerate(waypoints):
        D.loc[cell, :] = pd.Series( np.ravel(dists[i]), 
            index=data.index[dists[i].index])[data.index]
    end =time.time()
    print('Time for shortest paths: {} minutes'.format((end - start)/60))


    # ###############################################
    # Determine the perspective matrix
    # Distance matrix
    print('Determining perspectives, trajectory...')
    # Waypoint weights
    W = _weighting_scheme( D, voting_scheme )

    # Initalize trajectory to start cell distances
    trajectory = D.loc[start_cell, :]
    converged = False

    # Iteratively update perspective and determine trajectory
    iteration = 1
    while not converged and iteration < max_iterations:
        # Perspective matrix by alinging to start distances
        P = deepcopy(D)
        for wp in waypoints[1:]:
            # Position of waypoints relative to start
            idx_val = trajectory[wp]
            
            # Convert all cells before starting point to the negative
            before_indices = trajectory.index[ trajectory < idx_val ]
            P.loc[wp, before_indices] = -D.loc[wp, before_indices]

            # Align to start
            P.loc[wp, :] = P.loc[wp, :] + idx_val

        # Weighted trajectory
        new_traj = P.multiply(W).sum()

        # Check for convergence
        corr = pearsonr( trajectory, new_traj )[0]
        print('Correlation at iteration %d: %.4f' % (iteration, corr))
        if corr > 0.9999:
            converged = True

        # If not converged, continue iteration
        trajectory = new_traj
        iteration += 1

    # Terminal states
    print('Determining terminal states...')
    cells, _ = _stationary_distribution(data, waypoints, knn, n_jobs, trajectory)

    # Entropy and branch probabilities
    print('Entropy and branch probabilities...')
    ent, branch_probs = _differentiation_entropy(data, waypoints, knn, n_jobs, start_cell, trajectory, cells)
    
    # Add terminals
    ent = ent.append(pd.Series(0, index=cells))[trajectory.index]
    bp = pd.DataFrame(0, index=cells, columns=cells)
    bp.values[range(len(cells)), range(len(cells))] = 1
    branch_probs = branch_probs.append(bp.loc[:, branch_probs.columns])
    branch_probs = branch_probs.loc[trajectory.index, :]

    # Project results to all cells
    print('Project results to all cells...')
    ent = pd.Series(np.ravel(np.dot(
        W.T, ent.loc[W.index].values.reshape(-1, 1)) ), index=W.columns)
    branch_probs = pd.DataFrame(np.dot(
        W.T, branch_probs.loc[W.index, :]), index=W.columns, columns=branch_probs.columns)

    # UPdate results into dictionary
    res = {}
    res['waypoints'] = waypoints
    res['entropy'] = ent
    res['branch_probs'] = branch_probs
    res['trajectory'] = trajectory
        
    return res



















def _flock(cls, i, data, IDX, nbrs):
        med_data = np.median(data[IDX[i,:],:],axis=0)
        return nbrs.kneighbors(med_data.reshape(1, -1), n_neighbors=1, return_distance=False)[0][0]

def flock_waypoints(data_, waypoints_, waypoints_dim, knn, n_jobs, flock=2):

    if flock == 0:
        return

    # Nearest neighbors
    data = data_.values
    nbrs = NearestNeighbors(n_neighbors=knn, 
            metric='euclidean', n_jobs=n_jobs).fit(data) 

    # Flock
    waypoints_ = np.where( data_.index.isin( waypoints_ ) )[0]
    for f in range(flock):
        IDX = nbrs.kneighbors([data[i, :] for i in waypoints], return_distance=False)
        waypoints = Parallel(n_jobs=n_jobs)(
            delayed(_flock)(i, data, IDX, nbrs) for i in range(len(waypoints)))

    # Remove duplicates
    waypoints = data_.columns[waypoints]
    waypoints_dim = waypoints_dim.loc[~waypoints.duplicated()]
    waypoints = waypoints[~waypoints.duplicated()]
    waypoints_dim.index = waypoints
    
    return waypoints, waypoints_dim



def _determine_end_points(data, waypoints, knn, n_jobs, trajectory, start_cell):

    # Construct kNN graph of the waypoints
    print('Clustering...')
    # KMeans clustering        
    n_clusters = data.shape[1] * 2
    labels = pd.Series( KMeans(n_clusters = n_clusters).fit_predict(data.loc[waypoints, :]),
        index=waypoints)

    # Nearest neighbor graph
    nn = NearestNeighbors(n_neighbors=knn, 
            metric='euclidean', n_jobs=n_jobs).fit(data.loc[waypoints, :]) 
    adj = nn.kneighbors_graph(data.loc[waypoints, :], mode='distance')

    # Boundary cells
    boundary_cells = data.loc[waypoints].idxmin().append(
        data.loc[waypoints].idxmax()).unique()
    N = len(waypoints)

    # Remove outgoing edges
    x, y, z = find(adj)
    use_inds = np.where(np.ravel(trajectory[waypoints[x]]) < 
        np.ravel(trajectory[waypoints[y]]))[0]
    adj = csr_matrix( (z[use_inds], (x[use_inds], y[use_inds])), shape=[N, N] )


    # Cluster connectivity graph
    print('Boundary cell paths and connectivity graph')
    # Shortest paths between all boundary points
    wp_graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
    nx.relabel_nodes( wp_graph, 
        dict(zip(range(N), waypoints)), copy=False)

    # Paths between all end cells
    paths = []
    for i in range(len(boundary_cells)):
        for j in range(len(boundary_cells)):
            try:
                paths = paths + [nx.dijkstra_path(wp_graph, boundary_cells[i], boundary_cells[j])]
            except nx.NetworkXNoPath:
                pass

    # Cluster graph
    # Traverse through the paths adding directional edges
    cluster_order = trajectory[waypoints].groupby(labels).mean()
    cluster_graph = nx.DiGraph()
    for path in paths:
        # Add graph edges when labels are different
        inds = np.where(np.ravel(labels[path][:-1]) != np.ravel(labels[path][1:]))[0]
        for ind in inds:
            src = labels[path[ind]]
            tgt = labels[path[ind + 1]]
            if cluster_order[src] < cluster_order[tgt]:
                cluster_graph.add_edge( src, tgt )
            else:
                cluster_graph.add_edge( tgt, src )
    wp_paths = paths

    # Clusters with no outgoing edges are end points
    end_clusters = pd.Series([len(cluster_graph.neighbors(i)) 
        for i in cluster_graph.nodes()], index=cluster_graph.nodes())
    end_clusters = end_clusters.index[end_clusters == 0]
    # Corresponding waypoints
    end_points = []
    for c in end_clusters:
        end_points = end_points + [ trajectory[labels.index[labels == c]].idxmax() ]

    # Multiple paths through the graph
    cluster_paths = dict()
    for c in end_points:
        paths = []

        path = nx.dijkstra_path(cluster_graph, labels[start_cell], labels[c])
        paths = paths + [tuple(path)]

        # Clusters along the path
        # Remove each node to check presence of alternative path
        for cluster in path:
            if cluster == labels[start_cell] or cluster == labels[c]:
                continue
            temp = deepcopy(cluster_graph)
            temp.remove_node(cluster)
            try:
                alt_path = nx.dijkstra_path( temp, labels[start_cell], labels[c])
            except nx.NetworkXNoPath:
                # No alternative path
                continue

            # Path found, make sure it is distinct to existing path
            if len(set(alt_path).difference(path) ) > 0:
                paths = paths + [tuple(alt_path)]
        cluster_paths[c] = set(paths)


    # Remove end_points from from labels
    labels = labels[labels.index.difference(end_points)]
    return end_points, cluster_paths, labels, wp_paths, cluster_graph
