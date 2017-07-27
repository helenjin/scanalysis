import numpy as np
import pandas as pd


def filter_scseq_data(df, filter_cell_min=0, filter_cell_max=0, filter_gene_nonzero=None, filter_gene_mols=None):
    """
    Filter single cell RNA-seq data

    :parameter: df, which is a pandas DataFrame object
    :return: data, a pandas DataFrame object
    """
    
    if filter_cell_min != filter_cell_max:
        sums = df.sum(axis=1)
        to_keep = np.intersect1d(np.where(sums >= filter_cell_min)[0],
                                 np.where(sums <= filter_cell_max)[0])
        data = df.ix[df.index[to_keep], :].astype(np.float32)

    if filter_gene_nonzero != None:
        nonzero = df.astype(bool).sum(axis=0)
        to_keep = np.where(nonzero >= filter_gene_nonzero)[0]
        data = df.ix[:, to_keep].astype(np.float32)
        
    if filter_gene_mols != None:
        sums = df.sum(axis=0)
        to_keep = np.where(sums >= filter_gene_mols)[0]
        data = df.ix[:, to_keep].astype(np.float32)

    print("Successfully filtered data")
    return data

def normalize_scseq_data(df):
    """
    Normalize single cell RNA-seq data: Divide each cell by its molecule count
    and multiply counts of cells by the median of the molecule counts
    :parameter: df, a pandas DataFrame object
    :return: data, a pandas DataFrame object
    """

    molecule_counts = df.sum(axis=1)
    data = df.div(molecule_counts, axis=0) \
        .mul(np.median(molecule_counts), axis=0)

    # check that none of the genes are empty; if so remove them
    nonzero_genes = data.sum(axis=0) != 0
    data = data.ix[:, nonzero_genes].astype(np.float32)

    # set unnormalized_cell_sums

    print("Successfully normalized data")
    return data
