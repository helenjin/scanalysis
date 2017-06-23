import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # catch experimental ipython widget warning
    import seaborn as sns

def get_fig(fig=None, ax=None, figsize=[6.5, 6.5]):
    """fills in any missing axis or figure with the currently active one
    :param ax: matplotlib Axis object
    :param fig: matplotlib Figure object
    """
    if not fig:
        fig = plt.figure(figsize=figsize)
    if not ax:
        ax = plt.gca()
    return fig, ax

def plot_pca_variance_explained(data, n_components=30,
            fig=None, ax=None, ylim=(0, 0.1)):
        """ Plot the variance explained by different principal components
        :param n_components: Number of components to show the variance
        :param ylim: y-axis limits
        :param fig: matplotlib Figure object
        :param ax: matplotlib Axis object
        :return: fig, ax
        """
#        if self.pca is None:
#            raise RuntimeError('Please run run_pca() before plotting')

        fig, ax = get_fig(fig=fig, ax=ax)
        ax.plot(np.ravel(data['eigenvalues'].values))
        plt.ylim(ylim)
        plt.xlim((0, n_components))
        plt.xlabel('Components')
        plt.ylabel('Variance explained')
        plt.title('Principal components')
        sns.despine(ax=ax)
        return fig, ax
