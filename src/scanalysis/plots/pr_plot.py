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


with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # catch experimental ipython widget warning
    sns.set(context="paper", style='ticks', font_scale=1.5, font='Bitstream Vera Sans')
cmap = matplotlib.cm.Spectral_r
size = 8

def qualitative_colors(n):
    """ Generalte list of colors
    :param n: Number of colors
    """
    return sns.color_palette('Set1', n)


class DiffEntrResults(object):
	"""
	Container of multibranch results
	"""
	# Set up Rgam
	rgam = importr('gam')

	def __init__(self, trajectory, branches, branch_prob, no_bins=500):

		# Initialize
		self._trajectory = (trajectory - trajectory.min()) / (trajectory.max() - trajectory.min())
		self._trajectory = self._trajectory.sort_values()
		self._branches = branches
		self._branch_prob = branch_prob
		self._branch_prob[self._branch_prob < 0.01] = 0
		self._traj_bins = np.linspace(np.min(self.trajectory), np.max(self.trajectory), no_bins)

		self.branch_colors = dict( zip([2, 1, 3], qualitative_colors(3)))
                                                       


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
		#marker_trends, marker_std = self.compute_marker_trends(marker_data, branches, True, n_jobs=1)

		marker_trends = self.compute_marker_trends(marker_data, branches)
		marker_std = deepcopy(marker_trends)
        
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

def plot_palantir_on_tsne(DiffEntrResults, tsne):
    """ Plot Wishbone results on tSNE maps
    """
    
    input("Please make sure that the tSNE data entered corresponds to the DiffEntrResults object you've entered.\n\
    If yes, press enter to continue.\n\
    If not, Ctrl-C to exit and retry with correct parameters.")
    
    # Please run Wishbone using run_wishbone before plotting #
    # Please run tSNE before plotting #

    # Set up figure
    fig = plt.figure(figsize=[8, 4])
    gs = plt.GridSpec(1, 2)

    # Trajectory
    ax = plt.subplot(gs[0, 0])
    plt.scatter(tsne['x'], tsne['y'],
        edgecolors='none', s=size, cmap=cmap, c=DiffEntrResults.trajectory)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.title('DiffEntrResults trajectory')
    
    if DiffEntrResults.branches is None:
        DiffEntrResults.branches = DiffEntrResults.branch_prob.columns
    # Branch
    if DiffEntrResults.branches is not None:
        s = True
        if s:
            ax = plt.subplot(gs[0, 1])
            plt.scatter(tsne['x'], tsne['y'],
                edgecolors='none', s=size, 
                color=[DiffEntrResults.branch_colors[i] for i in DiffEntrResults.branches])
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            plt.title('Branch associations')
    
    return fig, ax