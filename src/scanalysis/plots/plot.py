import warnings
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import font_manager
from scipy.stats import gaussian_kde
from cycler import cycler
from sklearn.decomposition import PCA

try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # catch experimental ipython widget warning
    import seaborn as sns
    sns.set(context="paper", style='ticks', font_scale=1.5, font='Bitstream Vera Sans')
    
# set plotting defaults
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # catch warnings that system can't find fonts
    import seaborn as sns
    fm = font_manager.fontManager
    fm.findfont('Raleway')
    fm.findfont('Lato')
    
cmap = matplotlib.cm.Spectral_r
size = 8


class FigureGrid:
    """
    Generates a grid of axes for plotting
    axes can be iterated over or selected by number. e.g.:
    >>> # iterate over axes and plot some nonsense
    >>> fig = FigureGrid(4, max_cols=2)
    >>> for i, ax in enumerate(fig):
    >>>     plt.plot(np.arange(10) * i)
    >>> # select axis using indexing
    >>> ax3 = fig[3]
    >>> ax3.set_title("I'm axis 3")
    """
    
    
    # Figure Grid is favorable for displaying multiple graphs side by side.

    def __init__(self, n: int, max_cols=3, scale=3):
        """
        :param n: number of axes to generate
        :param max_cols: maximum number of axes in a given row
        """

        self.n = n
        self.nrows = int(np.ceil(n / max_cols))
        self.ncols = int(min((max_cols, n)))
        figsize = self.ncols * scale, self.nrows * scale

        # create figure
        self.gs = plt.GridSpec(nrows=self.nrows, ncols=self.ncols)
        self.figure = plt.figure(figsize=figsize)

        # create axes
        self.axes = {}
        for i in range(n):
            row = int(i // self.ncols)
            col = int(i % self.ncols)
            self.axes[i] = plt.subplot(self.gs[row, col])

    def __getitem__(self, item):
        return self.axes[item]

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def tight_layout(self, **kwargs):
        """wrapper for plt.tight_layout"""
        self.gs.tight_layout(self.figure, **kwargs)

    def despine(self, top=True, right=True, bottom=False, left=False):
        """removes axis spines (default=remove top and right)"""
        despine(ax=self, top=top, right=right, bottom=bottom, left=left)

    def detick(self, x=True, y=True):
        """
        removes tick labels
        :param x: bool, if True, remove tick labels from x-axis
        :param y: bool, if True, remove tick labels from y-axis
        """

        for ax in self:
            detick(ax, x=x, y=y)

    def savefig(self, filename, pad_inches=0.1, bbox_inches='tight', *args, **kwargs):
        """
        wrapper for savefig, including necessary paramters to avoid cut-off
        :param filename: str, name of output file
        :param pad_inches: float, number of inches to pad
        :param bbox_inches: str, method to use when considering bbox inches
        :param args: additional args for plt.savefig()
        :param kwargs: additional kwargs for plt.savefig()
        :return:
        """
        self.figure.savefig(
            filename, pad_inches=pad_inches, bbox_inches=bbox_inches, *args, **kwargs)

        
def qualitative_colors(n):
    """ Generate list of colors
    :param n: Number of colors
    """
    return sns.color_palette('Set1', n)


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

def density_2d(x, y):
    """return x and y and their density z, sorted by their density (smallest to largest)
    :param x:
    :param y:
    :return:
    """
    xy = np.vstack([np.ravel(x), np.ravel(y)])
    z = gaussian_kde(xy)(xy)
    i = np.argsort(z)
    return np.ravel(x)[i], np.ravel(y)[i], np.arcsinh(z[i])


def plot_molecules_per_cell_and_gene(data, fig=None, ax=None):

    height = 4
    width = 12
    fig = plt.figure(figsize=[width, height])
    gs = plt.GridSpec(1, 3)
    colsum = np.log10(data.sum(axis=0))
    rowsum = np.log10(data.sum(axis=1))
    for i in range(3):
        ax = plt.subplot(gs[0, i])

        if i == 0:
            print(np.min(rowsum))
            print(np.max(rowsum))
            n, bins, patches = ax.hist(rowsum, bins='auto')
            plt.xlabel('Molecules per cell (log10 scale)')
        elif i == 1:
            temp = np.log10(data.astype(bool).sum(axis=0))
            n, bins, patches = ax.hist(temp, bins='auto')
            plt.xlabel('Nonzero cells per gene (log10 scale)')
        else:
            n, bins, patches = ax.hist(colsum, bins='auto') 
            plt.xlabel('Molecules per gene (log10 scale)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        ax.tick_params(axis='x', labelsize=8)

    return fig, ax


## IF I have time, change the run_pca() to the one in this package.
def plot_pca_variance_explained_v1(data, n_components=30,
        fig=None, ax=None, ylim=(0, 0.1), random=True):
    """ Plot the variance explained by different principal components
    :param n_components: Number of components to show the variance
    :param ylim: y-axis limits
    :param fig: matplotlib Figure object
    :param ax: matplotlib Axis object
    :return: fig, ax

    from Wishbone package, revised
    """
    
    solver = 'randomized'
    if random != True:
        solver = 'full'
    pca = PCA(n_components=n_components, svd_solver=solver)
    pca.fit(data.values)

    fig, ax = get_fig(fig=fig, ax=ax)
    ax.plot(np.ravel(pca.explained_variance_ratio_))
    plt.ylim(ylim)
    plt.xlim((0, n_components))
    plt.xlabel('Components')
    plt.ylabel('Variance explained')
    plt.title('Principal components')
    sns.despine(ax=ax)
    return fig, ax
    
def plot_pca_variance_explained_v2(data, n_components=30,
        fig=None, ax=None, ylim=(0, 100), random=True):
    """ Plot the variance explained by different principal components
    :param n_components: Number of components to show the variance
    :param ylim: y-axis limits
    :param fig: matplotlib Figure object
    :param ax: matplotlib Axis object
    :return: fig, ax

    from MAGIC package
    """

    solver = 'randomized'
    if random != True:
        solver = 'full'
    pca = PCA(n_components=n_components, svd_solver=solver)
    pca.fit(data.values)

    fig, ax = get_fig(fig=fig, ax=ax)
    plt.plot(np.multiply(np.cumsum(pca.explained_variance_ratio_), 100))
    plt.ylim(ylim)
    plt.xlim((0, n_components))
    plt.xlabel('Number of principal components')
    plt.ylabel('% explained variance')
    return fig, ax


def plot_tsne(tsne, fig=None, ax=None, title='tSNE projection'):
    """Plot tSNE projections of the data
    :param fig: matplotlib Figure object
    :param ax: matplotlib Axis object
    :param title: Title for the plot
    """
    if tsne is None:
        raise RuntimeError('Please run tSNE using run_tsne before plotting ')
    fig, ax = get_fig(fig=fig, ax=ax)
    plt.scatter(tsne['x'], tsne['y'], s=size,
        color=qualitative_colors(2)[1])
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_title(title)
    return fig, ax


def plot_tsne_by_cell_sizes(data, tsne, fig=None, ax=None, vmin=None, vmax=None):
    """Plot tSNE projections of the data with cells colored by molecule counts
    :param fig: matplotlib Figure object
    :param ax: matplotlib Axis object
    :param vmin: Minimum molecule count for plotting
    :param vmax: Maximum molecule count for plotting
    :param title: Title for the plot
    """
#    if self.data_type == 'masscyt':
#        raise RuntimeError( 'plot_tsne_by_cell_sizes is not applicable \n\
#            for mass cytometry data. ' )

    fig, ax = get_fig(fig, ax)
    if tsne is None:
        raise RuntimeError('Please run run_tsne() before plotting.')
#    if self._normalized:
#        sizes = self.library_sizes
    else:
        sizes = data.sum(axis=1)
    plt.scatter(tsne['x'], tsne['y'], s=size, c=sizes, cmap=cmap)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.colorbar()
    return fig, ax

def plot_gene_expression(data, tsne, genes):
    """ Plot gene expression on tSNE maps
    :param genes: Iterable of strings to plot on tSNE
    """

    not_in_dataframe = set(genes).difference(data.columns)
    if not_in_dataframe:
        if len(not_in_dataframe) < len(genes):
            print('The following genes were either not observed in the experiment, '
                  'or the wrong gene symbol was used: {!r}'.format(not_in_dataframe))
        else:
            print('None of the listed genes were observed in the experiment, or the '
                  'wrong symbols were used.')
            return

    # remove genes missing from experiment
    genes = set(genes).difference(not_in_dataframe)

    height = int(2 * np.ceil(len(genes) / 5))
    width = 10
    fig = plt.figure(figsize=[width, height+0.25])
    n_rows = int(height / 2)
    n_cols = int(width / 2)
    gs = plt.GridSpec(n_rows, n_cols)

    axes = []
    for i, g in enumerate(genes):
        ax = plt.subplot(gs[i // n_cols, i % n_cols])
        axes.append(ax)
      #  if self.data_type == 'sc-seq':
        plt.scatter(tsne['x'], tsne['y'], c=np.arcsinh(data[g]),
                    cmap=cmap, edgecolors='none', s=size)
       # else:
       #     plt.scatter(self.tsne['x'], self.tsne['y'], c=self.data[g],
       #                 cmap=cmap, edgecolors='none', s=size)
        ax.set_title(g)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    return fig, axes

def plot_diffusion_components(tsne, diffusion_eigenvectors, diffusion_eigenvalues, title='Diffusion Components'):
    """ Plots the diffusion components on tSNE maps
    :return: fig, ax
    """
   # if self.tsne is None:
   #     raise RuntimeError('Please run tSNE before plotting diffusion components.')
   # if self.diffusion_eigenvectors is None:
   #     raise RuntimeError('Please run diffusion maps using run_diffusion_map before plotting')

    height = int(2 * np.ceil(diffusion_eigenvalues.shape[0] / 5))
    width = 10
    fig = plt.figure(figsize=[width, height])
    n_rows = int(height / 2)
    n_cols = int(width / 2)
    gs = plt.GridSpec(n_rows, n_cols)

    for i in range(diffusion_eigenvectors.shape[1]):
        ax = plt.subplot(gs[i // n_cols, i % n_cols])
        plt.scatter(tsne['x'], tsne['y'], c=diffusion_eigenvectors[i],
                    cmap=cmap, edgecolors='none', s=size)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_aspect('equal')
        plt.title( 'Component %d' % i, fontsize=10 )

    # fig.suptitle(title, fontsize=12)
    return fig, ax

def _correlation(x: np.array, vals: np.array):
    x = x[:, np.newaxis]
    mu_x = x.mean()  # cells
    mu_vals = vals.mean(axis=0)  # cells by gene --> cells by genes
    sigma_x = x.std()
    sigma_vals = vals.std(axis=0)

    return ((vals * x).mean(axis=0) - mu_vals * mu_x) / (sigma_vals * sigma_x)
    
def run_diffusion_map_correlations(data, diffusion_eigenvectors, components=None, no_cells=10):
    """ Determine gene expression correlations along diffusion components
    :param components: List of components to generate the correlations. All the components
    are used by default.
    :param no_cells: Window size for smoothing
    :return: None
    """
#    if self.data_type == 'masscyt':
#        raise RuntimeError('This function is designed to work for single cell RNA-seq')
#    if self.diffusion_eigenvectors is None:
#        raise RuntimeError('Please run diffusion maps using run_diffusion_map before determining correlations')

    if components is None:
        components = np.arange(diffusion_eigenvectors.shape[1])
    else:
        components = np.array(components)
    components = components[components != 0]

    # Container
    diffusion_map_correlations = np.empty((data.shape[1],
                                           diffusion_eigenvectors.shape[1]),
                                           dtype=np.float)
    for component_index in components:
        component_data = diffusion_eigenvectors.ix[:, component_index]

        order = data.index[np.argsort(component_data)]
        x = component_data[order].rolling(no_cells).mean()[no_cells:]
        # x = pd.rolling_mean(component_data[order], no_cells)[no_cells:]

        # this fancy indexing will copy self.data
        vals = data.ix[order, :].rolling(no_cells).mean()[no_cells:].values
        # vals = pd.rolling_mean(self.data.ix[order, :], no_cells, axis=0)[no_cells:]
        cor_res = _correlation(x, vals)
        # assert cor_res.shape == (gene_shape,)
        diffusion_map_correlations[:, component_index] = _correlation(x, vals)

    # this is sorted by order, need it in original order (reverse the sort)
#    self.diffusion_map_correlations = pd.DataFrame(diffusion_map_correlations[:, components],
#                        index=self.data.columns, columns=components)
    res = pd.DataFrame(diffusion_map_correlations[:, components],
                       index=data.columns, columns=components)
    return res

def plot_gene_component_correlations(diffusion_map_correlations
        , components=None, fig=None, ax=None,
        title='Gene vs. Diffusion Component Correlations'):
    """ plots gene-component correlations for a subset of components
    :param components: Iterable of integer component numbers
    :param fig: Figure
    :param ax: Axis
    :param title: str, title for the plot
    :return: fig, ax
    """
    fig, ax = get_fig(fig=fig, ax=ax)
    if diffusion_map_correlations is None:
        raise RuntimeError('Please run determine_gene_diffusion_correlations() '
                           'before attempting to visualize the correlations.')

    if components is None:
        components = diffusion_map_correlations.columns
    colors = qualitative_colors(len(components))

    for c,color in zip(components, colors):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # catch experimental ipython widget warning
            sns.kdeplot(diffusion_map_correlations[c].fillna(0), label=c,
                        ax=ax, color=color)
    sns.despine(ax=ax)
    ax.set_title(title)
    ax.set_xlabel('correlation')
    ax.set_ylabel('gene density')
    plt.legend()
    return fig, ax

###
def scatter_gene_expression(data, genes, density=False, color=None, fig=None, ax=None):
    """ 2D or 3D scatter plot of expression of selected genes
    :param genes: Iterable of strings to scatter
    """

    not_in_dataframe = set(genes).difference(data.columns.get_level_values(0))
    if not_in_dataframe:
        if len(not_in_dataframe) < len(genes):
            print('The following genes were either not observed in the experiment, '
                  'or the wrong gene symbol was used: {!r}'.format(not_in_dataframe))
        else:
            print('None of the listed genes were observed in the experiment, or the '
                  'wrong symbols were used.')
        return

    if len(genes) < 2 or len(genes) > 3:
        raise RuntimeError('Please specify either 2 or 3 genes to scatter.')

    for i in range(len(genes)):
        genes[i] = data.columns.values[np.where([genes[i] in col for col in data.columns.values])[0]][0]

    gui_3d_flag = True
    if ax == None:
        gui_3d_flag = False

    fig, ax = get_fig(fig=fig, ax=ax)
    if len(genes) == 2:
        if density == True:
            # Calculate the point density
            xy = np.vstack([data[genes[0]], data[genes[1]]])
            z = gaussian_kde(xy)(xy)

            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x, y, z = data[genes[0]][idx], data[genes[1]][idx], z[idx]

            plt.scatter(x, y, s=size, c=z, edgecolors='none')
            ax.set_title('Color = density')
            plt.colorbar()
        elif isinstance(color, pd.Series):
            plt.scatter(data[genes[0]], data[genes[1]],
                        s=size, c=color, edgecolors='none')
            if isinstance(color, str):
                ax.set_title('Color = ' + color.name)
            plt.colorbar()
        elif color in data.columns.get_level_values(0):
            color = data.columns.values[np.where([color in col for col in data.columns.values])[0]][0]
            plt.scatter(data[genes[0]], data[genes[1]],
                        s=size, c=data[color], edgecolors='none')
            ax.set_title('Color = ' + color)
            plt.colorbar()
        else:
            plt.scatter(data[genes[0]], data[genes[1]], edgecolors='none',
                        s=size, color=qualitative_colors(2)[1] if color == None else color)
        ax.set_xlabel(genes[0][1])
        ax.set_ylabel(genes[1][1])

    else:
        if not gui_3d_flag:
            ax = fig.add_subplot(111, projection='3d')

        if density == True:
            xyz = np.vstack([data[genes[0]],data[genes[1]],
                             data[genes[2]]])
            kde = gaussian_kde(xyz)
            density = kde(xyz)

            p = ax.scatter(data[genes[0]], data[genes[1]], data[genes[2]],
                       s=size, c=density, edgecolors='none')
            ax.set_title('Color = density')
            fig.colorbar(p)
        elif isinstance(color, pd.Series):
            p = ax.scatter(data[genes[0]], data[genes[1]],
                       data[genes[2]], s=size, c=color, edgecolors='none')
            ax.set_title('Color = ' + color.name)
            fig.colorbar(p)
        elif color in data.columns.get_level_values(0):
            color = data.columns.values[np.where([color in col for col in data.columns.values])[0]][0]
            p = ax.scatter(data[genes[0]], data[genes[1]],
                       data[genes[2]], s=size, c=data[color], edgecolors='none')
            ax.set_title('Color = ' + color)
            fig.colorbar(p)
        else:
            p = ax.scatter(data[genes[0]], data[genes[1]], data[genes[2]], 
                       edgecolors='none', s=size, color=qualitative_colors(2)[1] if color == None else color)
        ax.set_xlabel(genes[0][1])
        ax.set_ylabel(genes[1][1])
        ax.set_zlabel(genes[2][1])
        ax.view_init(15,55)
    
    plt.axis('tight')
    plt.tight_layout()
    return fig, ax

def savefig(fig, filename, pad_inches=0.1, bbox_inches='tight', *args, **kwargs):
        """
        wrapper for savefig, including necessary paramters to avoid cut-off
        :param filename: str, name of output file
        :param pad_inches: float, number of inches to pad
        :param bbox_inches: str, method to use when considering bbox inches
        :param args: additional args for plt.savefig()
        :param kwargs: additional kwargs for plt.savefig()
        :return:
        """
        fig.savefig(
            filename, pad_inches=pad_inches, bbox_inches=bbox_inches, *args, **kwargs)
