SCAnalysis
--------------

single cell analysis package
combines MAGIC and Wishbone, with additional features (including Palantir)


Wishbone is an algorithm to align single cells from differentiation systems with bifurcating branches. Wishbone has been designed to work with multidimensional single cell data from diverse technologies such as Mass cytometry and single cell RNA-seq.

#### Installation and dependencies
1. SCAnalysis has been implemented in Python3 and can be installed using

        $> git clone git://github.com/helenjin/scanalysis.git
        $> cd scanalysis
        $> sudo pip3 install .

2. SCAnalysis depends on a number of `python3` packages available on pypi and these dependencies are listed in `setup.py`
All the dependencies will be automatically installed using the above commands

#### Usage
##### Interactive command line
A tutorial on SCAnalysis usage and results visualization for single cell RNA-seq data can be found in this notebook: 
https://nbviewer.jupyter.org/github/helenjin/scanalysis/blob/master/notebooks/SCAnalysis.ipynb


#### Citations
Setty M, Tadmor MD, Reich-Zeliger S, Angel O, Salame TM, Kathail P, Choi K, Bendall S, Friedman N, Pe’er D. "Wishbone identifies bifurcating developmental trajectories from single-cell data." Nat. Biotech. 2016 April 12. <http://dx.doi.org/10.1038/nbt.3569>

van Dijk, David, et al. "MAGIC: A diffusion-based imputation method reveals gene-gene interactions in single-cell RNA-sequencing data." BioRxiv (2017): 111591. <http://www.biorxiv.org/content/early/2017/02/25/111591>