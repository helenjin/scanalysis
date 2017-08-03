"""
A python object representing single cell data
"""
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




def __init__(self, scdata, dm_eigs, start_cell, num_waypoints,
    knn=25, flock=2, n_jobs=1, voting_scheme='exponential', max_iterations=25):

    # Initialize
    self.scdata = scdata
    self.knn = knn
    self.n_jobs = n_jobs
    self.num_waypoints = num_waypoints
    self.flock = flock
    self.voting_scheme = voting_scheme
    self.max_iterations = max_iterations

    # Multi scale distance
    eig_vals = np.ravel(self.scdata.diffusion_eigenvalues.values[dm_eigs])
    self.data = self.scdata.diffusion_eigenvalues.values[:, dm_eigs] * (eig_vals / (1-eig_vals))
    self.data = pd.DataFrame( self.data, 
        index=self.scdata.data.columns, columns=dm_eigs )

    # Find start cell
    self.start_cell = random.sample( start_cell  , 1)[0]

    


# Max min sampling of waypoints
def max_min_sampling(self):
    
    waypoint_set, waypoint_sampling_dim = Multibranch._max_min_worker(self.data, self.num_waypoints)

    # Update object
    self.waypoints = waypoint_set
    self.waypoints_dim = waypoint_sampling_dim
    self.waypoints_dim = pd.Series(self.waypoints_dim, index=self.waypoints)

    # Flock waypoints if specified
    self.flock_waypoints( flock=self.flock )


def _max_min_worker(cls, data, num_waypoints):
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
def _weighting_scheme(cls, D, voting_scheme):
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




def _stationary_distribution(self):
    # kNN graph 
    n_neighbors = self.knn
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean',
        n_jobs=self.n_jobs).fit(self.data.loc[self.waypoints,:])
    # nbrs = NearestNeighbors(n_neighbors=self.knn, metric='cosine', algorithm='brute',
    #     n_jobs=1).fit(self.data.loc[self.waypoints,:])
    kNN = nbrs.kneighbors_graph(self.data.loc[self.waypoints,:], mode='distance' ) 
    dist,ind = nbrs.kneighbors(self.data.loc[self.waypoints,:])

    # Standard deviation allowing for "back" edges
    adaptive_std = np.ravel(dist[:, int(np.floor(n_neighbors / 3)) - 1])

    # Directed graph construction
    # Trajectory position of all the neighbors
    traj_nbrs = pd.DataFrame(self.trajectory[np.ravel(self.waypoints[ind])].values.reshape( 
        [len(self.waypoints), n_neighbors]), index=self.waypoints)

    # Remove edges that move backwards in trajectory except for edges that are within 
    # the computed standard deviation
    rem_edges = traj_nbrs.apply(lambda x : x < self.trajectory[traj_nbrs.index] - adaptive_std )
    rem_edges = rem_edges.stack()[rem_edges.stack()]

    # Determine the indices and update adjacency matrix
    cell_mapping = pd.Series(range(len(self.waypoints)), index=self.waypoints)
    x = list(cell_mapping[rem_edges.index.get_level_values(0)])
    y = list(rem_edges.index.get_level_values(1)) 
    # Update adjacecy matrix
    kNN[x, ind[x, y]] = 0  

    # Affinity matrix and markov chain
    x, y, z = find(kNN)
    aff = np.exp(-(z ** 2)/(adaptive_std[x] ** 2)  * 0.5 \
         -(z ** 2)/(adaptive_std[y] ** 2)  * 0.5 )
    W = csr_matrix((aff, (x, y)), [len(self.waypoints), len(self.waypoints)])
    # W = csr_matrix((1-z, (x, y)), [len(self.waypoints), len(self.waypoints)])

    # Probabilities
    D = np.ravel(W.sum(axis = 1))
    x, y, z = find(W)
    T = csr_matrix(( z / D[x], (x, y)), [len(self.waypoints), len(self.waypoints)])

    from scipy.sparse.linalg import eigs
    vals, vecs = eigs(T.T, 10)

    ranks = np.abs(np.real(vecs[:, np.argsort(vals)[-1]]))
    ranks = pd.Series(ranks, index=self.waypoints)

    # Cutoff and intersection with the boundary cells
    cutoff = norm.ppf(0.9999, loc=np.median(ranks), 
        scale=np.median(np.abs((ranks - np.median(ranks)))))
    dm_boundaries = pd.Index(set(self.data.idxmax()).union(self.data.idxmin()))
    cells = (ranks.index[ ranks > cutoff].intersection( dm_boundaries ))

    # Clusters cells
    Z = hierarchy.linkage(pairwise_distances(self.data.loc[cells,:]))
    clusters = pd.Series(hierarchy.fcluster(Z, 1, 'distance'), index=cells)  

    # IDentify cells with maximum trajectory for each cluster
    cells = self.trajectory[clusters.index].groupby(clusters).idxmax()

    return cells, ranks



def _differentiation_entropy(self, boundary_cells=None):

    # Identify boundary cells
    if boundary_cells is None:
        print('Identifying end points...')
        res = self._determine_end_points()
        boundary_cells = res[0]

    # Markov chain construction
    print('Markov chain construction...')

    # kNN graph 
    n_neighbors = self.knn
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean',
        n_jobs=self.n_jobs).fit(self.data.loc[self.waypoints,:])
    # nbrs = NearestNeighbors(n_neighbors=self.knn, metric='cosine', algorithm='brute',
    #     n_jobs=1).fit(self.data.loc[self.waypoints,:])
    kNN = nbrs.kneighbors_graph(self.data.loc[self.waypoints,:], mode='distance' ) 
    dist,ind = nbrs.kneighbors(self.data.loc[self.waypoints,:])

    # Standard deviation allowing for "back" edges
    adaptive_std = np.ravel(dist[:, int(np.floor(n_neighbors / 3)) - 1])

    # Directed graph construction
    # Trajectory position of all the neighbors
    traj_nbrs = pd.DataFrame(self.trajectory[np.ravel(self.waypoints[ind])].values.reshape( 
        [len(self.waypoints), n_neighbors]), index=self.waypoints)

    # Remove edges that move backwards in trajectory except for edges that are within 
    # the computed standard deviation
    rem_edges = traj_nbrs.apply(lambda x : x < self.trajectory[traj_nbrs.index] - adaptive_std )
    rem_edges = rem_edges.stack()[rem_edges.stack()]

    # Determine the indices and update adjacency matrix
    cell_mapping = pd.Series(range(len(self.waypoints)), index=self.waypoints)
    x = list(cell_mapping[rem_edges.index.get_level_values(0)])
    y = list(rem_edges.index.get_level_values(1)) 
    # Update adjacecy matrix
    kNN[x, ind[x, y]] = 0  

    # Affinity matrix and markov chain
    x, y, z = find(kNN)
    aff = np.exp(-(z ** 2)/(adaptive_std[x] ** 2)  * 0.5 \
         -(z ** 2)/(adaptive_std[y] ** 2)  * 0.5 )
    W = csr_matrix((aff, (x, y)), [len(self.waypoints), len(self.waypoints)])
    # W = csr_matrix((1-z, (x, y)), [len(self.waypoints), len(self.waypoints)])

    # Probabilities
    D = np.ravel(W.sum(axis = 1))
    x, y, z = find(W)
    T = csr_matrix(( z / D[x], (x, y)), [len(self.waypoints), len(self.waypoints)])


    # Absorption states should not have outgoing edges
    abs_states = np.where(self.waypoints.isin(boundary_cells))[0]
    # Reset absorption state affinities by Removing neigbors
    T[abs_states,:] = 0 
    # Diagnoals as 1s
    T[abs_states, abs_states] = 1



    # Fundamental matrix and absorption probabilities
    print('Computing fundamental matrix and absorption probabilities...')
    # Transition states
    trans_states = list(set(range(len(self.waypoints))).difference(abs_states))

    # Q matrix
    Q = T[trans_states,:][:, trans_states]
    # Fundamental matrix
    mat = np.eye(Q.shape[0]) - Q.todense()
    N = inv(mat)

    # Absorption probabilities
    abs_probabilities = np.dot(N, T[trans_states,:][:, abs_states].todense())
    abs_probabilities = pd.DataFrame(abs_probabilities, 
        index=self.waypoints[trans_states],
        columns=self.waypoints[abs_states])
    abs_probabilities[abs_probabilities < 0] = 0

    # Entropy
    ent = abs_probabilities.apply(entropy, axis=1)

    return ent, abs_probabilities


def _shortest_path_helper(cls, cell, adj):
    # NOTE: Graph construction is parallelized since constructing the graph outside was creating lock issues
    graph = nx.Graph(adj)
    return pd.Series(nx.single_source_dijkstra_path_length(graph, cell))

def run_multibranch(self):

    # # ################################################
    # # Sample waypoints
    print('Sampling and flocking waypoints...')
    start =time.time()

    # Append start cell
    if isinstance(self.num_waypoints, int):
        self.max_min_sampling(  )
        self.waypoints = pd.Index(self.waypoints.difference([self.start_cell]).unique())
    else:
        self.waypoints = self.num_waypoints
    self.waypoints = pd.Index([self.start_cell]).append(self.waypoints)
    end =time.time()
    print('Time for determining waypoints: {} minutes'.format((end - start)/60))


    # ################################################
    # Shortest path distances to determine trajectories
    print('Shortest path distances...')
    start =time.time()
    nbrs = NearestNeighbors(n_neighbors=self.knn, 
            metric='euclidean', n_jobs=self.n_jobs).fit(self.data) 
    adj = nbrs.kneighbors_graph(self.data, mode='distance')

    # Distances
    dists = Parallel(n_jobs=self.n_jobs)(
        delayed(self._shortest_path_helper)(np.where(self.data.index == cell)[0][0], adj) 
            for cell in self.waypoints)

    # Convert to distance matrix
    D = pd.DataFrame(0.0, index=self.waypoints, columns=self.data.index)
    for i, cell in enumerate(self.waypoints):
        D.loc[cell, :] = pd.Series( np.ravel(dists[i]), 
            index=self.data.index[dists[i].index])[self.data.index]
    end =time.time()
    print('Time for shortest paths: {} minutes'.format((end - start)/60))


    # ###############################################
    # Determine the perspective matrix
    # Distance matrix
    print('Determining perspectives, trajectory...')
    # Waypoint weights
    W = Multibranch._weighting_scheme( D, self.voting_scheme )

    # Initalize trajectory to start cell distances
    trajectory = D.loc[self.start_cell, :]
    converged = False

    # Iteratively update perspective and determine trajectory
    iteration = 1
    while not converged and iteration < self.max_iterations:
        # Perspective matrix by alinging to start distances
        P = deepcopy(D)
        for wp in self.waypoints[1:]:
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

    self.trajectory = trajectory
    # return

    # Terminal states
    print('Determining terminal states...')
    cells, _ = self._stationary_distribution()

    # Entropy and branch probabilities
    print('Entropy and branch probabilities...')
    ent, branch_probs = self._differentiation_entropy(cells)
    # Add terminals
    ent = ent.append(pd.Series(0, index=cells))[self.trajectory.index]
    bp = pd.DataFrame(0, index=cells, columns=cells)
    bp.values[range(len(cells)), range(len(cells))] = 1
    branch_probs = branch_probs.append(bp.loc[:, branch_probs.columns])
    branch_probs = branch_probs.loc[self.trajectory.index, :]

    # Project results to all cells
    print('Project results to all cells...')
    ent = pd.Series(np.ravel(np.dot(
        W.T, ent.loc[W.index].values.reshape(-1, 1)) ), index=W.columns)
    branch_probs = pd.DataFrame(np.dot(
        W.T, branch_probs.loc[W.index, :]), index=W.columns, columns=branch_probs.columns)

    # UPdate object
    self.entropy = ent
    self.branch_probs = branch_probs
    
























































































def _determine_branches(self, D, P):

    # ########### Disagreement matrix and spectral clustering
    print('Clustering of the disagreement matrix...')
    Q = np.abs(P.ix[self.waypoints, self.waypoints] - D.ix[self.start_cell, self.waypoints])
    QD = pd.DataFrame(pairwise_distances(Q), index=Q.index, columns=Q.columns)
    
    # Affinities and laplacian
    std = np.std(np.ravel(QD))
    A = np.exp(-QD / std)
    # Convert to affinities using local scaling
    Dinv = np.ravel(A.sum())
    Dinv[Dinv!=0] = 1/Dinv[Dinv!=0]
    L = pd.DataFrame(np.diag(Dinv).dot(A), index=A.index, columns=A.columns)
    
    # Number of clustesr
    EV, ED = GraphDiffusion.graph_diffusion.GetEigs(L.values, L.shape[0]-2, None, 1)
    no_clusters = np.where(ED < 0.01)[0][0]
    print('Number of clusters: %d' % no_clusters)

    # # Cluster
    while True:
        labels = SpectralClustering(n_clusters=no_clusters, assign_labels='discretize',
            affinity='precomputed').fit_predict(A)
        labels = pd.Series(labels, index=self.waypoints)
        counts = pd.Series( Counter(labels) )
        if sum(counts < 5) == 0:
            break
        no_clusters -= sum( counts < 5)
    print('Number of clusters: %d' % no_clusters)


    # self.branches = labels
    # # Branch colors
    # colors = pd.Series( list(sns.color_palette('hls', 
    #     len(set(self.branches)))), index=set(self.branches))
    # self.branch_colors = pd.Series(['#%02X%02X%02X' % (int(i[0]*255),
    #         int(i[1]*255), int(i[2]*255)) for i in colors], index=colors.index)
    # return


    # ######### Directed graph from clusters
    # Boundary points
    print('Paths between boundary points...')
    boundary_cells = self.data.ix[self.waypoints].idxmin().append(
        self.data.ix[self.waypoints].idxmax()).unique()


    # Paths between boundary points
    # Nearest neighbor graph
    knn = 15
    nn = NearestNeighbors(n_neighbors=knn, 
            metric='euclidean', n_jobs=self.n_jobs).fit(self.data.ix[labels.index]) 
    adj = nn.kneighbors_graph(self.data.ix[labels.index], mode='distance')
    
    # Networkx graph
    x, y, z = find(adj)
    df = pd.DataFrame(columns=['x', 'y', 'weight'])
    df['x'] = labels.index[x]; df['y'] = labels.index[y]; df['weight'] = z
    wp_graph = nx.from_pandas_dataframe(df, 'x', 'y', ['weight'])            

    # Paths between all pairs of boundary points
    paths = []
    # Pairwise paths
    for i in range(len(boundary_cells)-1):
        for j in range(i+1, len(boundary_cells)):
            paths = paths + [nx.dijkstra_path(wp_graph, boundary_cells[i], boundary_cells[j])]

    for i in range(len(boundary_cells)):
        paths = paths + [nx.dijkstra_path(wp_graph, self.start_cell, boundary_cells[i])]

    # Cluster order
    cluster_order = self.trajectory[labels.index].groupby(labels).median()

    # Cluster connectivity
    print('Branches and branch connectivity...')
    prev_graph = nx.DiGraph(); reduced_graph = nx.DiGraph()

    # Repeat until graph structure doesn not change
    while (not nx.is_isomorphic(prev_graph, reduced_graph)) or (len(reduced_graph.nodes()) == 0):

        # Store results from the current iteration
        prev_graph = deepcopy(reduced_graph)
        cluster_conn = nx.Graph()
        for path in paths:
            # Add graph edges when labels are different
            inds = np.where(np.ravel(labels[path][:-1]) != np.ravel(labels[path][1:]))[0]
            
            for ind in inds:
                src = path[ind]
                tgt = path[ind + 1]
                weight = wp_graph.edge[src][tgt]['weight']
                if cluster_conn.has_edge(labels[src], labels[tgt]):
                    weight = np.min([weight, cluster_conn.edge[labels[src]][labels[tgt]]['weight']])

                # Edge direction
                # if self.trajectory[src] <= self.trajectory[tgt]:
                if cluster_order[labels[src]] <= cluster_order[labels[tgt]]:
                    edge_direction = (labels[src], labels[tgt])
                    wp = [src, tgt]
                else:
                    edge_direction = (labels[tgt], labels[src])
                    wp = [tgt, src]
                cluster_conn.add_edge( labels[src], labels[tgt], 
                    weight=weight, edge_direction=edge_direction, wp = wp)


        # Remove triangles
        edges = np.array(cluster_conn.edges(data=True))
        edges = edges[np.argsort([i[2]['weight'] for i in cluster_conn.edges(data=True)])][::-1]
        for x, y, _ in edges:
            # Remove edge if x and y share neighbors
            if len(set(cluster_conn.neighbors(x)).intersection(cluster_conn.neighbors(y))):
                cluster_conn.remove_edge(x, y)

        
        # Directed graph
        temp = nx.DiGraph()
        for i, j in cluster_conn.edges():
            x, y = cluster_conn[i][j]['edge_direction']
            temp.add_edge( x, y,  
                weight=cluster_conn[i][j]['weight'], wp=cluster_conn[i][j]['wp'])
        cluster_conn = deepcopy(temp)

        # Reduce graph
        # Combine nodes which have only one child and one parent
        reduced_graph = deepcopy(cluster_conn)
        labels = labels[labels.isin(cluster_conn.nodes())]
        cluster_order = self.trajectory[labels.index].groupby(labels).mean()
        branches = deepcopy(labels)
        for node in cluster_order.sort_values().index[::-1]:
            if len(reduced_graph.out_edges(node)) == 1:# and len(reduced_graph.in_edges(node)) == 1:
                child = reduced_graph.out_edges(node)[0][1]
                
                # Merge parent and child if the child has no other parents
                if len(reduced_graph.in_edges(child)) == 1:
                    cells = branches.index[branches == child]
                    branches[cells] = node

                    # Remove child and change edges
                    for child_edge in reduced_graph.out_edges(child):
                        reduced_graph.add_edge(node, child_edge[1])

                    reduced_graph.remove_node(child)


        # Save for next iteration
        labels = deepcopy(branches)


    # Save results
    branches = branches.astype(np.int64)
    self.branch_conn = reduced_graph
    self.branches = branches

    # # Adjust branch points
    # print('Adjusting branch points...')
    # updated_branches = deepcopy(branches)
    # for node in self.branch_conn.nodes():

    #     # Diverging
    #     if len(self.branch_conn.neighbors(node)) > 1 :
    #         new_branches = self._adjust_branch_point(node, 
    #             self.branch_conn.neighbors(node), diverging=True)
    #         cells = self.branches.index[self.branches == node]
    #         updated_branches[cells] = new_branches[cells]

    #     # Converging
    #     if len(self.branch_conn.predecessors(node)) > 1 :
    #         new_branches = self._adjust_branch_point(node, 
    #             self.branch_conn.predecessors(node), diverging=False)
    #         cells = self.branches.index[self.branches == node]
    #         updated_branches[cells] = new_branches[cells]
    # self.branches = updated_branches




def _adjust_branch_point(self, test_branch, nbr_branches, nw=5, diverging=True):


    # Set directions based on diverging (branch) OR converging (loop)
    nbr_range = range(0, nw)
    if diverging:
        nbr_update = 0
    else:
        nbr_update = -1
            
    # Build branch dictionary with sorted order of waypoints
    branch_dict = pd.Series(index=set(self.branches))
    for b in set(self.branches):
        branch_dict[b] = self.trajectory[self.branches.index[self.branches == b]].sort_values().index
    
    
    # Construct graph with all the relevant cells
    sub = self.branches.index[self.branches.isin( nbr_branches + [test_branch] )]
    nn = NearestNeighbors(n_neighbors=10, 
            metric='euclidean', n_jobs=self.n_jobs).fit(self.data.ix[sub])
    adj = nn.kneighbors_graph(self.data.ix[sub], mode='distance')
    
    # Networkx graph for shortest paths
    x, y, z = find(adj)
    df = pd.DataFrame(columns=['x', 'y', 'weight'])
    df['x'] = sub[x]; df['y'] = sub[y]; df['weight'] = z
    g = nx.from_pandas_dataframe(df, 'x', 'y', ['weight'])    
    
    
    # Find shortest paths between child nodes and find the point closest to the start
    closest_points = []
    for i,c1 in enumerate(nbr_branches[:len(nbr_branches)-1]):
        for c2 in nbr_branches[(i+1):]:
            
            # Paths between pairs of waypoints
            for c1_w in branch_dict[c1][nbr_range]:
                for c2_w in branch_dict[c2][nbr_range]:

                    # Shortest path
                    path = nx.dijkstra_path(g, c1_w, c2_w)
                    # Minimum trajectory
                    if diverging:
                        closest_points = closest_points + [np.min(self.trajectory[path])]                    
                    else:
                        closest_points = closest_points + [np.max(self.trajectory[path])]

                    

    # Find shorest distances to the child node that is equidistant to the median cell
    # Cells to be adjusted
    if diverging:
        cells = branch_dict[test_branch][ self.trajectory[branch_dict[test_branch]] >= np.median(closest_points) ]
    else:
        cells = branch_dict[test_branch][ self.trajectory[branch_dict[test_branch]] <= np.median(closest_points) ]

    # If not cells are to bed adjusted
    if len(cells) == 0:
        return self.branches

    if diverging:
        median_cell = self.trajectory[cells].idxmin()
    else:
        median_cell = self.trajectory[cells].idxmax()

    cells = cells.difference([median_cell])

    # Find equidistant cells in all branches relative to the median cell to adjust branches
    dists_to_median = pd.Series([[]] * len(nbr_branches), index = nbr_branches)
    for nbr in nbr_branches:
        m_dists = nx.single_source_dijkstra_path_length(g, median_cell)
        m_dists = pd.Series(list(m_dists.values()), index=list(m_dists.keys()))
        dists_to_median[nbr] = m_dists[branch_dict[nbr]].sort_values()

    # Maximum of the min dists for equidistant cells
    # mm_dist = np.max(dists_to_median.apply(lambda x: x[0]))
    # equi_cells  = dists_to_median.apply(lambda x: 
    #     x.index[np.where((np.ravel(x[:-1]) <= mm_dist) & (np.ravel(x[1:]) >= mm_dist))[0][0]])
    mm_dists = dists_to_median.apply(lambda x: x[0]).sort_values()[::-1]
    for mm_dist in mm_dists:
        equi_cells  = dists_to_median.apply(lambda x: 
            np.where((np.ravel(x[:-1]) <= mm_dist) & (np.ravel(x[1:]) >= mm_dist))[0])

        # Skip branches if they are too far
        covered_nodes = equi_cells.index[equi_cells.apply(len) == 0]
        if len(covered_nodes) > 0:
            dists_to_median = dists_to_median[covered_nodes]
        else:
            equi_cells  = dists_to_median.apply(lambda x: 
                x.index[np.where((np.ravel(x[:-1]) <= mm_dist) & (np.ravel(x[1:]) >= mm_dist))[0][0]])
            break

    # Update branches
    # Return values
    updated_branches = deepcopy(self.branches)
    cell_dists = pd.DataFrame( index=cells, columns=equi_cells.index )
    for nbr in equi_cells.index:
        temp = pd.Series( nx.single_source_dijkstra_path_length(g, equi_cells[nbr]) )
        cell_dists.ix[cells, nbr] = temp[cells]
    updated_branches[cells] = cell_dists.idxmin( axis=1 )

    return updated_branches

# Function to add waypoints
def _add_waypoints(self, add_waypoints=10):

    # Initalize
    waypoints = pd.Series()
    waypoints_dim = pd.Series()

    print('Identifying additional waypoints...')
    for edge in self.branch_conn.edges(data=True):
        # Waypoints connecting edges
        e_wp = self.trajectory[edge[2]['wp']].sort_values().index

        # Cells in the range
        cells = self.trajectory.index[(self.trajectory >= self.trajectory[e_wp[0]]) & \
            (self.trajectory <= self.trajectory[e_wp[1]])]
        data = self.data.ix[cells, self.waypoints_dim[e_wp]]
        data = data.iloc[:, ~data.columns.duplicated()]

        # Determine waypoints
        w, s = Multibranch._max_min_worker( data,  add_waypoints)
        waypoints = waypoints.append( w.to_series() )
        waypoints_dim = waypoints_dim.append( s )

    # Update object
    self.waypoints = self.waypoints.append( waypoints )
    self.waypoints_dim = self.waypoints_dim.append( waypoints_dim)
    print('Flocking...')
    self.flock_waypoints(self.flock)


# Functions for flocking waypoints
def _flock(cls, i, data, IDX, nbrs):
    med_data = np.median(data[IDX[i,:],:],axis=0)
    return nbrs.kneighbors(med_data.reshape(1, -1), n_neighbors=1, return_distance=False)[0][0]

def flock_waypoints(self, flock=2):

    if flock == 0:
        return

    # Nearest neighbors
    data = self.data.values
    nbrs = NearestNeighbors(n_neighbors=self.knn, 
            metric='euclidean', n_jobs=self.n_jobs).fit(data) 

    # Flock
    waypoints = np.where( self.data.index.isin( self.waypoints ) )[0]
    for f in range(flock):
        IDX = nbrs.kneighbors([data[i, :] for i in waypoints], return_distance=False)
        waypoints = Parallel(n_jobs=self.n_jobs)(
            delayed(Multibranch._flock)(i, data, IDX, nbrs) for i in range(len(waypoints)))

    # Remove duplicates
    waypoints = self.scdata.data.columns[waypoints]
    self.waypoints_dim = self.waypoints_dim.loc[~waypoints.duplicated()]
    self.waypoints = waypoints[~waypoints.duplicated()]
    self.waypoints_dim.index = self.waypoints


def _determine_end_points_prev(self):

    # Construct kNN graph of the waypoints
    print('Clustering...')
    nn = NearestNeighbors(n_neighbors=self.knn, 
            metric='euclidean', n_jobs=self.n_jobs).fit(self.data.loc[self.waypoints, :]) 
    adj = nn.kneighbors_graph(self.data.loc[self.waypoints, :], mode='distance')
    # dist,ind = nn.kneighbors(self.data.loc[self.waypoints, :])

    # Boundary cells
    boundary_cells = self.data.loc[self.waypoints].idxmin().append(
        self.data.loc[self.waypoints].idxmax()).unique()
    N = len(self.waypoints)
    labels = pd.Series( KMeans(n_clusters = self.data.shape[1] * 2).fit_predict(self.data.loc[self.waypoints, :]),
        index=self.waypoints)

    # Cluster connectivity graph
    print('Boundary cell paths and connectivity graph')
    # Shortest paths between all boundary points
    wp_graph = nx.from_scipy_sparse_matrix(adj)
    nx.relabel_nodes( wp_graph, 
        dict(zip(range(N), self.waypoints)), copy=False)
    paths = []
    for i in range(len(boundary_cells)-1):
        for j in range(i+1, len(boundary_cells)):
            paths = paths + [nx.dijkstra_path(wp_graph, boundary_cells[i], boundary_cells[j])]


    # Cluster graph
    # Traverse through the paths adding directional edges
    cluster_order = self.trajectory[labels.index].groupby(labels).median()
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


    # Clusters with no outgoing edges are end points
    end_clusters = pd.Series([len(cluster_graph.neighbors(i)) 
        for i in cluster_graph.nodes()], index=cluster_graph.nodes())
    end_clusters = end_clusters.index[end_clusters == 0]
    # Corresponding waypoints
    end_points = []
    for c in end_clusters:
        end_points = end_points + [ self.trajectory[labels.index[labels == c]].idxmax() ]


    # Remove end_points from from labels
    labels = labels[labels.index.difference(end_points)]
    return end_points, labels, cluster_graph

def  _page_rank(self):
    # kNN graph 
    n_neighbors = self.knn
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean',
        n_jobs=self.n_jobs).fit(self.data.loc[self.waypoints,:])
    # nbrs = NearestNeighbors(n_neighbors=self.knn, metric='cosine', algorithm='brute',
    #     n_jobs=1).fit(self.data.loc[self.waypoints,:])
    kNN = nbrs.kneighbors_graph(self.data.loc[self.waypoints,:], mode='distance' ) 
    dist,ind = nbrs.kneighbors(self.data.loc[self.waypoints,:])

    # Standard deviation allowing for "back" edges
    adaptive_std = np.ravel(dist[:, int(np.floor(n_neighbors / 3)) - 1])

    # Directed graph construction
    # Trajectory position of all the neighbors
    traj_nbrs = pd.DataFrame(self.trajectory[np.ravel(self.waypoints[ind])].values.reshape( 
        [len(self.waypoints), n_neighbors]), index=self.waypoints)

    # Remove edges that move backwards in trajectory except for edges that are within 
    # the computed standard deviation
    rem_edges = traj_nbrs.apply(lambda x : x < self.trajectory[traj_nbrs.index] - adaptive_std )
    rem_edges = rem_edges.stack()[rem_edges.stack()]

    # Determine the indices and update adjacency matrix
    cell_mapping = pd.Series(range(len(self.waypoints)), index=self.waypoints)
    x = list(cell_mapping[rem_edges.index.get_level_values(0)])
    y = list(rem_edges.index.get_level_values(1)) 
    # Update adjacecy matrix
    kNN[x, ind[x, y]] = 0  

    # Convert to networkx graph
    graph = nx.from_scipy_sparse_matrix(kNN, create_using=nx.DiGraph())
    nx.relabel_nodes( graph, 
        dict(zip(range(len(self.waypoints)), self.waypoints)), copy=False)

    # Run page rank
    ranks = pd.Series(nx.pagerank(graph,
        personalization=self.trajectory[self.waypoints].to_dict()))

    # Cutoff and intersection with the boundary cells
    cutoff = norm.ppf(0.9999, loc=np.median(ranks), 
        scale=np.median(np.abs((ranks - np.median(ranks)))))
    # dm_boundaries = pd.Index(set(self.data.idxmax()).union(self.data.idxmin()))
    dm_boundaries = pd.Index(set(self.scdata.DMEigs.idxmax()).union(self.scdata.DMEigs.idxmin()))
    cells = (ranks.index[ ranks > cutoff].intersection( dm_boundaries ))

    return cells, ranks



def _determine_end_points(self):

    # Construct kNN graph of the waypoints
    print('Clustering...')
    # KMeans clustering        
    n_clusters = self.data.shape[1] * 2
    labels = pd.Series( KMeans(n_clusters = n_clusters).fit_predict(self.data.loc[self.waypoints, :]),
        index=self.waypoints)

    # Nearest neighbor graph
    nn = NearestNeighbors(n_neighbors=self.knn, 
            metric='euclidean', n_jobs=self.n_jobs).fit(self.data.loc[self.waypoints, :]) 
    adj = nn.kneighbors_graph(self.data.loc[self.waypoints, :], mode='distance')

    # Boundary cells
    boundary_cells = self.data.loc[self.waypoints].idxmin().append(
        self.data.loc[self.waypoints].idxmax()).unique()
    N = len(self.waypoints)

    # Remove outgoing edges
    x, y, z = find(adj)
    use_inds = np.where(np.ravel(self.trajectory[self.waypoints[x]]) < 
        np.ravel(self.trajectory[self.waypoints[y]]))[0]
    adj = csr_matrix( (z[use_inds], (x[use_inds], y[use_inds])), shape=[N, N] )


    # Cluster connectivity graph
    print('Boundary cell paths and connectivity graph')
    # Shortest paths between all boundary points
    wp_graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
    nx.relabel_nodes( wp_graph, 
        dict(zip(range(N), self.waypoints)), copy=False)

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
    cluster_order = self.trajectory[self.waypoints].groupby(labels).mean()
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
        end_points = end_points + [ self.trajectory[labels.index[labels == c]].idxmax() ]

    # Multiple paths through the graph
    cluster_paths = dict()
    for c in end_points:
        paths = []

        path = nx.dijkstra_path(cluster_graph, labels[self.start_cell], labels[c])
        paths = paths + [tuple(path)]

        # Clusters along the path
        # Remove each node to check presence of alternative path
        for cluster in path:
            if cluster == labels[self.start_cell] or cluster == labels[c]:
                continue
            temp = deepcopy(cluster_graph)
            temp.remove_node(cluster)
            try:
                alt_path = nx.dijkstra_path( temp, labels[self.start_cell], labels[c])
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
