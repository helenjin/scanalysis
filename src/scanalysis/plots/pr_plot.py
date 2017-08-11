import pandas as pd
import numpy as np
import networkx as nx
import pickle
from collections import OrderedDict
# import pygam
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2
from joblib import Parallel, delayed

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

import time
import tqdm

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr  
pandas2ri.activate()



matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['image.cmap'] = 'Spectral_r'
warnings.filterwarnings(action="ignore", module="matplotlib", message="findfont")
warnings.filterwarnings(action="ignore", module="pygam", message="divide by zero")


class DiffEntrResults(object):
	"""
	Container of multibranch results
	"""
	# # Set up Rgam
	# rgam = importr('gam')

	def __init__(self, trajectory, branches, branch_prob, branch_conn, no_bins=500):

		# Initialize
		self._trajectory = (trajectory - trajectory.min()) / (trajectory.max() - trajectory.min())
		self._trajectory = self._trajectory.sort_values()
		self._branches = branches
		self._branch_prob = branch_prob
		self._branch_prob[self._branch_prob < 0.01] = 0
		self._branch_conn = branch_conn
		self._traj_bins = np.linspace(np.min(self.trajectory), np.max(self.trajectory), no_bins)


	# Getters and setters
	@property
	def trajectory(self):
		return self._trajectory

	@trajectory.setter
	def trajectory(self, trajectory):
		self._trajectory = trajectory

	@property
	def branches(self):
		return self._branches

	@property
	def branch_prob(self):
		return self._branch_prob

	@property
	def branch_conn(self):
		return self._branch_conn

	@property 
	def traj_bins(self):
		return self._traj_bins

	@traj_bins.setter
	def traj_bins(self, traj_bins):
		self._traj_bins = traj_bins

	@classmethod
	def load(cls, pkl_file):
		with open(pkl_file, 'rb') as f:
			data = pickle.load(f)

		# Set up object
		mbr = cls(data['_trajectory'], data['_branches'], data['_branch_prob'], data['_branch_conn'])
		return mbr

	def save(self, pkl_file: str):
		pickle.dump(vars(self), pkl_filet)


	## ############# HELPER FUNCTIONS
	# Function to fit and GAM
	@classmethod
	def _gam_fit_predict(cls, x, y, weights=None, pred_x=None):

		# Weights
		if weights is None:
			weights = np.repeat(1.0, len(x))

		# Construct dataframe
		use_inds = np.where(weights > 0)[0]
		r_df = pandas2ri.py2ri(pd.DataFrame(np.array([x, y]).T[use_inds,:], columns=['x', 'y']))

		# Fit the model
		rgam = importr('gam')
		model = rgam.gam(Formula('y~s(x)'), data=r_df, weights=pd.Series(weights[use_inds]) )

		# Predictions
		if pred_x is None:
			pred_x = x
		y_pred = np.array(robjects.r.predict(model,
			newdata=pandas2ri.py2ri(pd.DataFrame(pred_x, columns=['x']))))

		deviance = np.array(robjects.r.deviance(model))
		vals = dict(zip(model.names, list(model)))
		df = vals['df.residual'][0]

		return y_pred, [deviance, df]

	## ############# ANALYSIS FUNCTIONS
	def _2gene_deviance_chi(self, expr_vals1, expr_vals2, weights1, weights2, standardize=True):

		# GAM Fit 1
		x1 = self.trajectory[expr_vals1.index].values
		y1 = expr_vals1.values

		# GAM Fit 2
		x2 = self.trajectory[expr_vals2.index].values
		y2 = expr_vals2.values

		if standardize:
			y1 = (y1-np.mean(y1))/np.std(y1)
			y2 = (y2-np.mean(y2))/np.std(y2)

		# Fit
		y_gam1, [deviance1, df1] = self._gam_fit_predict(x1, y1, weights1)
		y_gam2, [deviance2, df2] = self._gam_fit_predict(x2, y2, weights2)

		# GAM Fit Both
		x_both = np.append(x1, x2)
		y_both = np.append(y1, y2)
		y_gamboth, [devianceboth, dfboth] = self._gam_fit_predict(x_both, y_both, weights1.append(weights2))

		pvalue = 1 - chi2.cdf(devianceboth - (deviance1 + deviance2), dfboth - (df1+df2))

		return pvalue


	def _gene_likelihood_helper(self, expr_vals, weights):

		# GAM fit
		y_gam, [deviance, df] = self._gam_fit_predict(self.trajectory[expr_vals.index].values, expr_vals.values,
														  weights)

		# Intercept fit
		use_inds = np.where(weights > 0)[0]
		lin_intercept = LinearRegression()
		y_lin = lin_intercept.fit(np.repeat(1.0, len(use_inds)).reshape([-1, 1]), expr_vals.values[use_inds],
								  sample_weight=weights[use_inds]).predict(
			self.trajectory[expr_vals.index].values.reshape([-1, 1]))

		# Pvalue
		y = expr_vals.values
		pvalue = 1 - chi2.cdf(((y[use_inds] - y_lin[use_inds]) ** 2).sum() - deviance, len(use_inds) - 1 - df)

		return pvalue



	def gene_likelihood_test(self, marker_data, branches=None, n_jobs=-1):
		# Likelihood test to determine if a gene has a significant trend along the trajectory 
		if branches is None:
			branches = self.branch_prob.columns

		# p-value container
		p_vals = pd.DataFrame(1.0, index=marker_data.index, columns=self.branch_prob.columns)
		for branch in branches:
			print(branch)
			start = time.time()

			# Run in parallel
			weights = self.branch_prob.loc[marker_data.columns, branch]
			res = Parallel(n_jobs=n_jobs)(
				delayed(self._gene_likelihood_helper)(marker_data.loc[gene,:], weights)
				for gene in marker_data.index)

			# Update pvalues
			p_vals.loc[:, branch] = np.ravel(res)
			end = time.time()
			print('Time for processing {}: {} minutes'.format(branch, (end-start)/60))

		return p_vals


	def _marker_trends_helper(self, expr_vals, weights, compute_std):

		# GAM fit 
		y_gam, _ = self._gam_fit_predict(self.trajectory[expr_vals.index].values,
			expr_vals.values, weights, self.traj_bins)

		return y_gam


	def compute_marker_trends(self, marker_data, branches=None, compute_std=False, n_jobs=-1):

		# Link for computing standard errors of the predicted values
		# http://www.stat.wisc.edu/courses/st572-larget/Spring2007/handouts03-1.pdf

		# Compute for all branches if branch is not speicified
		if branches is None:
			branches = self.branch_prob.columns

		# Results Container
		marker_trends = OrderedDict()
		for branch in branches:
			marker_trends[branch] = pd.DataFrame(0.0, index=marker_data.index, columns=self.traj_bins)
		if compute_std:
			marker_std = deepcopy(marker_trends)

		# Compute for each branch
		for branch in branches:
			print(branch)
			start = time.time()

			# Branch cells and weights
			weights = self.branch_prob.loc[marker_data.columns, branch]
			res = Parallel(n_jobs=n_jobs)(
				delayed(self._marker_trends_helper)(marker_data.loc[gene,:], weights, compute_std)
				for gene in marker_data.index)

			# Fill in the matrices
			if not compute_std:
				marker_trends[branch].loc[:, :] = np.ravel(res).reshape([marker_data.shape[0], 
					len(self.traj_bins)])
			else:
				trends = [res[i][0] for i in range(len(res))]
				marker_trends[branch].loc[:, :] = np.ravel(trends).reshape([marker_data.shape[0], 
					len(self.traj_bins)])
				stds = [res[i][1] for i in range(len(res))]
				marker_std[branch].loc[:, :] = np.ravel(trends).reshape([marker_data.shape[0], 
					len(self.traj_bins)])
			end = time.time()
			print('Time for processing {}: {} minutes'.format(branch, (end-start)/60))

		# Adjust the boundary cases
		for branch in branches:
			br_cells = self.branch_prob.index[self.branch_prob.loc[:, branch] > 0 ]
			try:
				max_traj_bin = np.where(self.traj_bins > self.trajectory[br_cells].max())[0][0]

				# Adjust expression
				marker_trends[branch].iloc[:, max_traj_bin:] = \
					marker_trends[branch].iloc[:, max_traj_bin-1].values.reshape([marker_data.shape[0], 1])
				if compute_std:
					marker_std[branch].iloc[:, max_traj_bin:] = \
						marker_std[branch].iloc[:, max_traj_bin-1].values.reshape([marker_data.shape[0], 1])
			except IndexError:
				# Indicates branch extends to the end, so do nothing
				pass

		if compute_std:
			return marker_trends, marker_std
		return marker_trends




	def plot_markers(self, marker_data, branches=None, colors=None):

		# Marker trends and standard deviations
		if branches is None:
			branches = self.branch_prob.columns
		marker_trends, marker_std = self.compute_marker_trends(marker_data, branches, True, n_jobs=1)

		# Generate colors
		if colors is None:
			colors = sns.color_palette('hls', len(branches))
			colors = pd.Series(list(colors), index=branches)

		# Separate panel for each marker
		fig = plt.figure(figsize=[10, 4 * marker_data.shape[0]])
		for i, marker in enumerate(marker_data.index):
			ax = fig.add_subplot(marker_data.shape[0], 1, i+1)

			# Plot each branch 
			for branch in branches:
				# Trend
				means = marker_trends[branch].loc[marker, :]
				ax.plot( self.traj_bins, means, color=colors[branch], label=branch)

				# Standard deviation
				stds = marker_std[branch].loc[marker, :]
				plt.fill_between(self.traj_bins, means - stds,
					means + stds, alpha=0.1, color=colors[branch])
			ax.legend(loc=2, bbox_to_anchor=(1, 1), fontsize=12)
			ax.set_title(marker)
		sns.despine()
















































	# def _compute_marker_trend(X, y, predX):
	# 	X = X[X > 0]
	# 	y = y[X.index]
	# 	gam = pygam.LinearGAM().gridsearch(X.values.reshape([-1, 1]), y)
	# 	preds = gam.predict(predX.reshape([-1, 1]))
	# 	std = preds - gam.prediction_intervals(predX.reshape([-1, 1]), 0.68)[:, 0]
	# 	return preds, std





	def determine_branch_weights(self, no_bins=150, smoothing_factor=1, max_traj=None):

		# Sort trajectory
		trajectory = self.trajectory.copy()
		trajectory = trajectory.sort_values()

		# Normalize if necessary
		if max_traj is None:
			max_traj = np.max(trajectory)
		trajectory = trajectory - np.min(trajectory)
		trajectory = trajectory.divide( max_traj )

		# Compute bin locations and bin memberships
		bins = np.linspace(np.min(trajectory), np.max(trajectory), no_bins)

		# Compute gaussian weights for points at each location
		# Standard deviation estimated from Silverman's approximation
		print('Computing smoothing weights...')
		stdev = np.std(trajectory) * 1.06 * len(trajectory) **(-1/5) * smoothing_factor
		bin_membership = np.digitize(trajectory, bins)-1
		weights = np.exp(-((np.tile(bins[bin_membership], [no_bins, 1]).T - 
			bins) ** 2 / (2 * stdev**2))) * (1/(2*np.pi*stdev ** 2) ** 0.5) 
		weights = pd.DataFrame(weights, index=trajectory.index, columns=range(no_bins))


		# Branch weights are obtained by scaling by branch probabilities
		branch_weights = {}
		print('Computing branch specific weights...')
		for boundary in self.branch_prob.columns:
			print(boundary)

			probs = self.branch_prob.loc[:, boundary]

			# Remove any outliers
			br_max_traj = self.trajectory[self.branch_prob.loc[:, boundary] > 0.99].max()
			cells = self.branch_prob.index[self.branch_prob.loc[:, boundary] > 0]
			probs[ cells[self.trajectory[cells] > br_max_traj] ] = 0
			# 
			# Branch weights
			branch_weights[boundary] = weights.multiply(self.branch_prob.loc[:, boundary], axis=0)

		# Update object
		self.branch_weights = branch_weights
		self.traj_bins = bins
		self.trajectory = trajectory



	# Plotting functions
	# def plot_markers(self, marker_data, branch, scale=False):

		# markers = marker_data.index
		# cells = marker_data.columns.intersection(self.branch_prob.index)

		# # Determine the number of paths to the branch from start cluster
		# start_cluster =  pd.Series([len(self.branch_conn.predecessors(i)) 
		# 	for i in self.branch_conn.nodes()], index=self.branch_conn.nodes()).idxmin()
		# paths = list(nx.all_simple_paths(self.branch_conn, start_cluster, branch))


		# def find_trajs(p_cells):
		# 	weights = self.branch_weights[branch].loc[p_cells, :]
		# 	w_means = pd.DataFrame(np.dot( marker_data.loc[:, p_cells], 
		# 		weights ) / weights.sum().values, index=marker_data.index)

		# 	if scale:
		# 		w_means = w_means.subtract( w_means.min(axis=1), axis=0).\
		# 			divide( w_means.max(axis=1) - w_means.min(axis=1), axis=0)
		# 	return w_means


		# # Marker trajectory along each branch
		# path_trajs = dict()
		# path_cells = dict()
		# if len(paths) == 1:
		# 	path_trajs[0] = find_trajs(cells)
		# 	path_cells[0] = cells
		# else:
		# 	for src, tgt in zip([0, 1], [1, 0]):
		# 		src_path = paths[src]; tgt_path = paths[tgt]

		# 		# Find split point
		# 		for i in range(np.min([len(src_path), len(tgt_path)])):
		# 			if src_path[i] != tgt_path[i]:
		# 				break_point = tgt_path[i]
		# 				break
		# 		# Combine point
		# 		for i in range(np.min([len(src_path), len(tgt_path)])):
		# 			if src_path[::-1][i] != tgt_path[::-1][i]:
		# 				combine_point = tgt_path[::-1][i-1]
		# 				break

		# 		# Find branches to exclude
		# 		remove_clusters = nx.descendants(self.branch_conn, break_point).difference( 
		# 			nx.descendants(self.branch_conn, combine_point)) 
		# 		remove_clusters = remove_clusters.difference([combine_point]).union([break_point])

		# 		# Cells and trajectories
		# 		cells = self.branches.index[~self.branches.isin(remove_clusters)]
		# 		path_trajs[src] = find_trajs(cells)
		# 		path_cells[src] = cells


		# # # Marker trajectory along each branch
		# # branch_trajs = dict()
		# # for branch in self.branch_weights.keys():
		# # 	weights = self.branch_weights[branch].loc[cells, :]
		# # 	w_means = pd.DataFrame(np.dot( marker_data.loc[:, cells], 
		# # 		weights ) / weights.sum().values, index=marker_data.index)

		# # 	if scale:
		# # 		w_means = w_means.subtract( w_means.min(axis=1), axis=0).\
		# # 			divide( w_means.max(axis=1) - w_means.min(axis=1), axis=0)
		# # 	branch_trajs[branch] = w_means


		# # Marker colors
		# marker_colors = sns.color_palette('hls', len(markers))
		# fig = plt.figure(figsize=[8, 3 * len(path_trajs)])
		# gs = gridspec.GridSpec(len(path_trajs), 1)

		# for i in path_trajs.keys():
		# 	plt.subplot(gs[i, 0])

		# 	# Plot each branch
		# 	for marker, color in zip(markers, marker_colors):
		# 		plt.plot(self.traj_bins, path_trajs[i].loc[marker, :].values,
		# 			color=color, label=marker)


		# 		# Error clouds
		# 		cells = path_cells[i]

		# 		# Compute weighted standard deviation
		# 		# Differences to weighted mean for each bin
		# 		no_bins = len(self.traj_bins)
		# 		means = path_trajs[i].loc[marker, :].values
		# 		y = marker_data.loc[marker, cells]
		# 		if scale:
		# 			y = (y - y.min()) / (y.max() - y.min())
		# 		rep_mark = y.repeat(no_bins).values.reshape( [(len(cells)), no_bins] )
		# 		diffs = (rep_mark - means) ** 2

		# 		# Standard deviations.
		# 		weights = self.branch_weights[branch].loc[cells, :]
		# 		wstds = (rep_mark * weights).sum() / weights.sum()
		# 		# SCale
		# 		n_prime = (weights > 0).sum()
		# 		wstds = np.sqrt( wstds * n_prime / (n_prime - 1) ) * 0.5

		# 		# # Plot clouds
		# 		# plt.fill_between(self.traj_bins, means - wstds,
		# 		# 	means + wstds, alpha=0.1, color=color)


		# 	# Add legend
		# 	plt.legend(loc=2, bbox_to_anchor=(1, 1), fontsize=12)
		# return path_trajs 






