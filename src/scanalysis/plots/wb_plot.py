import warnings
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # catch experimental ipython widget warning
    import seaborn as sns
    
    
# set plotting defaults
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

class WBResults:

    def __init__(self, trajectory=None, waypoints=None, branch=None, bas=None, branch_colors=None):
        """
        Container class for single cell data
        :param data:  DataFrame of cells X genes representing expression
        :param data_type: Type of the data: Can be either 'sc-seq' or 'masscyt'
        :param metadata: None or DataFrame representing metadata about the cells
        """
        self.trajectory = trajectory
        self.waypoints = waypoints
        self.branch = branch
        self.bas = bas
        self.branch_colors = branch_colors


    def plot_wishbone_on_tsne(self, tsne):
        """ Plot Wishbone results on tSNE maps
        """
        
        input("Please make sure that the tSNE data entered corresponds to the Wishbone object you've entered.\n\
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
            edgecolors='none', s=size, cmap=cmap, c=self.trajectory)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.title('Wishbone trajectory')

        # Branch
        if self.branch is not None:
            s = True
            if s:
                ax = plt.subplot(gs[0, 1])
                plt.scatter(tsne['x'], tsne['y'],
                    edgecolors='none', s=size, 
                    color=[self.branch_colors[i] for i in self.branch])
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())
                plt.title('Branch associations')
        
        return fig, ax

    # Function to plot trajectory
    def plot_marker_trajectory(self, data, markers, show_variance=False,
        no_bins=150, smoothing_factor=1, min_delta=0.1, fig=None, ax=None):
        """Plot marker trends along trajectory
        :param markers: Iterable of markers/genes to be plotted.
        :param show_variance: Logical indicating if the trends should be accompanied with variance
        :param no_bins: Number of bins for calculating marker density
        :param smoothing_factor: Parameter controling the degree of smoothing
        :param min_delta: Minimum difference in marker expression after normalization to show separate trends for the two branches
        :param fig: matplotlib Figure object
        :param ax: matplotlib Axis object
        :return Dictionary containing the determined trends for the different branches
        """
        # Please run Wishbone run_wishbone before plotting #
        
        # Variance calculation is currently not supported for single-cell RNA-seq (sc-seq) #

        # Compute bin locations and bin memberships
        trajectory = self.trajectory.copy()
        # Sort trajectory
        trajectory = trajectory.sort_values()
        bins = np.linspace(np.min(trajectory), np.max(trajectory), no_bins)

        # Compute gaussian weights for points at each location
        # Standard deviation estimated from Silverman's approximation
        stdev = np.std(trajectory) * 1.34 * len(trajectory) **(-1/5) * smoothing_factor
        weights = np.exp(-((np.tile(trajectory, [no_bins, 1]).T -
            bins) ** 2 / (2 * stdev**2))) * (1/(2*np.pi*stdev ** 2) ** 0.5)


        # Adjust weights if data has branches
        if self.branch is not None:

            plot_branch = True

            # Branch of the trunk
            trunk = self.branch[trajectory.index[0]]
            branches = list( set( self.branch).difference([trunk]))
            linetypes = pd.Series([':', '--'], index=branches)


            # Counts of branch cells in each bin
            branch_counts = pd.DataFrame(np.zeros([len(bins)-1, 3]), columns=[1, 2, 3])
            for j in branch_counts.columns:
                branch_counts[j] = pd.Series([sum(self.branch[trajectory.index[(trajectory > bins[i-1]) & \
                    (trajectory < bins[i])]] == j) for i in range(1, len(bins))])
            # Frequencies
            branch_counts = branch_counts.divide( branch_counts.sum(axis=1), axis=0)

            # Identify the bin with the branch point by looking at the weights
            weights = pd.DataFrame(weights, index=trajectory.index, columns=range(no_bins))
            bp_bin = weights.columns[np.where(branch_counts[trunk] < 0.9)[0][0]] + 0
            if bp_bin < 0:
                bp_bin = 3

        else:
            plot_branch = False
            bp_bin = no_bins

        weights_copy = weights.copy()

        # Plot marker tsne_res
        xaxis = bins

        # Set up return object
        ret_values = dict()
        ret_values['Trunk'] = pd.DataFrame( xaxis[0:bp_bin], columns=['x'])
        ret_values['Branch1'] = pd.DataFrame( xaxis[(bp_bin-2):], columns=['x'])
        ret_values['Branch2'] = pd.DataFrame( xaxis[(bp_bin-2):], columns=['x'])

        # Marker colors
        colors = qualitative_colors( len(markers) )
        scaling_factor = 2
        linewidth = 3

        # Set up plot
        fig, ax = get_fig(fig, ax, figsize=[14, 4])

        for marker,color in zip(markers, colors):
           
            # Marker expression repeated no bins times
            y = data.ix[trajectory.index, marker]
            rep_mark = np.tile(y, [no_bins, 1]).T


            # Normalize y
            y_min = np.percentile(y, 1)
            y = (y - y_min)/(np.percentile(y, 99) - y_min)
            y[y < 0] = 0; y[y >  1] = 1;
            norm_rep_mark = pd.DataFrame(np.tile(y, [no_bins, 1])).T


            if not plot_branch:
                # Weight and plot
                vals = (rep_mark * weights)/sum(weights)

                # Normalize
                vals = vals.sum(axis=0)
                vals = vals - np.min(vals)
                vals = vals/np.max(vals)

                # Plot
                plt.plot(xaxis, vals, label=marker, color=color, linewidth=linewidth)

                # Show errors if specified
                if show_variance:

                    # Scale the marks based on y and values to be plotted
                    temp = (( norm_rep_mark - vals - np.min(y))/np.max(y)) ** 2
                    # Calculate standard deviations
                    wstds = inner1d(np.asarray(temp).T, np.asarray(weights).T) / weights.sum()

                    plt.fill_between(xaxis, vals - scaling_factor*wstds,
                        vals + scaling_factor*wstds, alpha=0.2, color=color)

                # Return values
                ret_values['Trunk'][marker] = vals[0:bp_bin]
                ret_values['Branch1'][marker] = vals[(bp_bin-2):]
                ret_values['Branch2'][marker] = vals[(bp_bin-2):]

            else: # Branching trajectory
                rep_mark = pd.DataFrame(rep_mark, index=trajectory.index, columns=range(no_bins))

                plot_split = True
                # Plot trunk first
                weights = weights_copy.copy()

                plot_vals = ((rep_mark * weights)/np.sum(weights)).sum()
                trunk_vals = plot_vals[0:bp_bin]

                branch_vals = []
                for br in branches:
                    # Mute weights of the branch cells and plot
                    weights = weights_copy.copy()
                    weights.ix[self.branch.index[self.branch == br], :] = 0

                    plot_vals = ((rep_mark * weights)/np.sum(weights)).sum()
                    branch_vals.append( plot_vals[(bp_bin-1):] )

                # Min and max
                temp = trunk_vals.append( branch_vals[0] ).append( branch_vals[1] )
                min_val = np.min(temp)
                max_val = np.max(temp)


                # Plot the trunk
                plot_vals = ((rep_mark * weights)/np.sum(weights)).sum()
                plot_vals = (plot_vals - min_val)/(max_val - min_val)
                plt.plot(xaxis[0:bp_bin], plot_vals[0:bp_bin],
                    label=marker, color=color, linewidth=linewidth)

                if show_variance:
                    # Calculate weighted stds for plotting
                    # Scale the marks based on y and values to be plotted
                    temp = (( norm_rep_mark - plot_vals - np.min(y))/np.max(y)) ** 2
                    # Calculate standard deviations
                    wstds = inner1d(np.asarray(temp).T, np.asarray(weights).T) / weights.sum()

                    # Plot
                    plt.fill_between(xaxis[0:bp_bin], plot_vals[0:bp_bin] - scaling_factor*wstds[0:bp_bin],
                        plot_vals[0:bp_bin] + scaling_factor*wstds[0:bp_bin], alpha=0.1, color=color)

                # Add values to return values
                ret_values['Trunk'][marker] = plot_vals[0:bp_bin]



                # Identify markers which need a split
                if sum( abs(pd.Series(branch_vals[0]) - pd.Series(branch_vals[1])) > min_delta ) < 5:
                    # Split not necessary, plot the trunk values
                    plt.plot(xaxis[(bp_bin-1):], plot_vals[(bp_bin-1):],
                        color=color, linewidth=linewidth)

                    # Add values to return values
                    ret_values['Branch1'][marker] = list(plot_vals[(bp_bin-2):])
                    ret_values['Branch2'][marker] = list(plot_vals[(bp_bin-2):])

                    if show_variance:
                        # Calculate weighted stds for plotting
                        # Scale the marks based on y and values to be plotted
                        temp = (( norm_rep_mark - plot_vals - np.min(y))/np.max(y)) ** 2
                        wstds = inner1d(np.asarray(temp).T, np.asarray(weights).T) / weights.sum()
                        # Plot
                        plt.fill_between(xaxis[(bp_bin-1):], plot_vals[(bp_bin-1):] - scaling_factor*wstds[(bp_bin-1):],
                            plot_vals[(bp_bin-1):] + scaling_factor*wstds[(bp_bin-1):], alpha=0.1, color=color)
                else:
                    # Plot the two branches separately
                    for br_ind,br in enumerate(branches):
                        # Mute weights of the branch cells and plot
                        weights = weights_copy.copy()

                        # Smooth weigths
                        smooth_bins = 10
                        if bp_bin < smooth_bins:
                            smooth_bins = bp_bin - 1
                        for i in range(smooth_bins):
                            weights.ix[self.branch == br, bp_bin + i - smooth_bins] *= ((smooth_bins - i)/smooth_bins) * 0.25
                        weights.ix[self.branch == br, (bp_bin):weights.shape[1]] = 0

                        # Calculate values to be plotted
                        plot_vals = ((rep_mark * weights)/np.sum(weights)).sum()
                        plot_vals = (plot_vals - min_val)/(max_val - min_val)
                        plt.plot(xaxis[(bp_bin-2):], plot_vals[(bp_bin-2):],
                            linetypes[br], color=color, linewidth=linewidth)

                        if show_variance:
                            # Calculate weighted stds for plotting
                            # Scale the marks based on y and values to be plotted
                            temp = (( norm_rep_mark - plot_vals - np.min(y))/np.max(y)) ** 2
                            # Calculate standard deviations
                            wstds = inner1d(np.asarray(temp).T, np.asarray(weights).T) / weights.sum()

                            # Plot
                            plt.fill_between(xaxis[(bp_bin-1):], plot_vals[(bp_bin-1):] - scaling_factor*wstds[(bp_bin-1):],
                                plot_vals[(bp_bin-1):] + scaling_factor*wstds[(bp_bin-1):], alpha=0.1, color=color)

                        # Add values to return values
                        ret_values['Branch%d' % (br_ind + 1)][marker] = list(plot_vals[(bp_bin-2):])


        # Clean up the plotting
        # Clean xlim
        plt.legend(loc=2, bbox_to_anchor=(1, 1), prop={'size':16})

        # Annotations
        # Add trajectory as underlay
        cm = matplotlib.cm.Spectral_r
        yval = plt.ylim()[0]
        yval = 0
        plt.scatter( trajectory, np.repeat(yval - 0.1, len(trajectory)),
            c=trajectory, cmap=cm, edgecolors='none', s=size)
        sns.despine()
        plt.xticks( np.arange(0, 1.1, 0.1) )

        # Clean xlim
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.2, 1.1 ])
        plt.xlabel('Wishbone trajectory')
        plt.ylabel('Normalized expression')
        
        return ret_values, fig, ax

    def plot_marker_heatmap(self, marker_trends, trajectory_range=[0, 1]):
        """ Plot expression of markers as a heatmap
        :param marker_trends: Output from the plot_marker_trajectory function
        :param trajectory_range: Range of the trajectory in which to plot the results
        """
        if trajectory_range[0] >= trajectory_range[1]:
            raise RuntimeError('Start cannot exceed end in trajectory_range')
        if trajectory_range[0] < 0 or trajectory_range[1] > 1:
            raise RuntimeError('Please use a range between (0, 1)')

        # Set up figure
        markers = marker_trends['Trunk'].columns[1:]

        if self.branch is not None:
            fig = plt.figure(figsize = [16, 0.5*len(markers)])
            gs = plt.GridSpec( 1, 2 )

            branches = np.sort(list(set(marker_trends.keys()).difference(['Trunk'])))
            for i,br in enumerate(branches):
                ax = plt.subplot( gs[0, i] )

                # Construct the full matrix
                mat = marker_trends['Trunk'].append( marker_trends[br][2:] )
                mat.index = range(mat.shape[0])

                # Start and end
                start = np.where(mat['x'] >= trajectory_range[0])[0][0]
                end = np.where(mat['x'] >= trajectory_range[1])[0][0]

                # Plot
                plot_mat = mat.ix[start:end]
                sns.heatmap(plot_mat[markers].T,
                    linecolor='none', cmap=cmap, vmin=0, vmax=1)
                ax.xaxis.set_major_locator(plt.NullLocator())
                ticks = np.arange(trajectory_range[0], trajectory_range[1]+0.1, 0.1)
                plt.xticks([np.where(plot_mat['x'] >= i)[0][0] for i in ticks], ticks)

                # Labels
                plt.xlabel( 'Wishbone trajectory' )
                plt.title( br )
        else:
            # Plot values from the trunk alone
            fig = plt.figure(figsize = [8, 0.5*len(markers)])
            ax = plt.gca()

            # Construct the full matrix
            mat = marker_trends['Trunk']
            mat.index = range(mat.shape[0])

            # Start and end
            start = np.where(mat['x'] >= trajectory_range[0])[0][0]
            end = np.where(mat['x'] >= trajectory_range[1])[0][0]

            # Plot
            plot_mat = mat.ix[start:end]
            sns.heatmap(plot_mat[markers].T,
                linecolor='none', cmap=cmap, vmin=0, vmax=1)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ticks = np.arange(trajectory_range[0], trajectory_range[1]+0.1, 0.1)
            plt.xticks([np.where(plot_mat['x'] >= i)[0][0] for i in ticks], ticks)

            # Labels
            plt.xlabel( 'Wishbone trajectory' )


        return fig, ax


    def plot_derivatives(self, marker_trends, trajectory_range=[0, 1]):
        """ Plot change in expression of markers along trajectory
        :param marker_trends: Output from the plot_marker_trajectory function
        :param trajectory_range: Range of the trajectory in which to plot the results
        """
        if trajectory_range[0] >= trajectory_range[1]:
            raise RuntimeError('Start cannot exceed end in trajectory_range')
        if trajectory_range[0] < 0 or trajectory_range[1] > 1:
            raise RuntimeError('Please use a range between (0, 1)')


        # Set up figure
        markers = marker_trends['Trunk'].columns[1:]

        if self.branch is not None:
            fig = plt.figure(figsize = [16, 0.5*len(markers)])
            gs = plt.GridSpec( 1, 2 )

            branches = np.sort(list(set(marker_trends.keys()).difference(['Trunk'])))
            for i,br in enumerate(branches):
                ax = plt.subplot( gs[0, i] )

                # Construct the full matrix
                mat = marker_trends['Trunk'].append( marker_trends[br][2:] )
                mat.index = range(mat.shape[0])

                # Start and end
                start = np.where(mat['x'] >= trajectory_range[0])[0][0]
                end = np.where(mat['x'] >= trajectory_range[1])[0][0]

                # Plot
                diffs = mat[markers].diff()
                diffs[diffs.isnull()] = 0

                # Update the branch points diffs
                bp_bin = marker_trends['Trunk'].shape[0]
                diffs.ix[bp_bin-1] = marker_trends[br].ix[0:1, markers].diff().ix[1]
                diffs.ix[bp_bin] = marker_trends[br].ix[1:2, markers].diff().ix[2]
                diffs = diffs.ix[start:end]
                mat = mat.ix[start:end]

                # Differences
                vmax = max(0.05,  abs(diffs).max().max() )
                # Plot
                sns.heatmap(diffs.T, linecolor='none',
                    cmap=matplotlib.cm.RdBu_r, vmin=-vmax, vmax=vmax)
                ax.xaxis.set_major_locator(plt.NullLocator())
                ticks = np.arange(trajectory_range[0], trajectory_range[1]+0.1, 0.1)
                plt.xticks([np.where(mat['x'] >= i)[0][0] for i in ticks], ticks)

                # Labels
                plt.xlabel( 'Wishbone trajectory' )
                plt.title( br )
        else:
            # Plot values from the trunk alone
            fig = plt.figure(figsize = [8, 0.5*len(markers)])
            ax = plt.gca()

            # Construct the full matrix
            mat = marker_trends['Trunk']
            mat.index = range(mat.shape[0])

            # Start and end
            start = np.where(mat['x'] >= trajectory_range[0])[0][0]
            end = np.where(mat['x'] >= trajectory_range[1])[0][0]

            # Plot
            diffs = mat[markers].diff()
            diffs[diffs.isnull()] = 0
            diffs = diffs.ix[start:end]
            mat = mat.ix[start:end]

            # Differences
            vmax = max(0.05,  abs(diffs).max().max() )
            # Plot
            sns.heatmap(diffs.T, linecolor='none',
                cmap=matplotlib.cm.RdBu_r, vmin=-vmax, vmax=vmax)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ticks = np.arange(trajectory_range[0], trajectory_range[1]+0.1, 0.1)
            plt.xticks([np.where(mat['x'] >= i)[0][0] for i in ticks], ticks)

            # Labels
            plt.xlabel( 'Wishbone trajectory' )

        return fig, ax
