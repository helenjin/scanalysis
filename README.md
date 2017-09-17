SCAnalysis
------

Single Cell Analysis package
combines MAGIC and Wishbone, with additional features (including Palantir)


* Wishbone is an algorithm to align single cells from differentiation systems with bifurcating branches. Wishbone has been designed to work with multidimensional single cell data from diverse technologies such as Mass cytometry and single cell RNA-seq. 

* MAGIC (Markov-Affinity Based Graph Imputation of Cells) is an interactive tool to impute missing values in single-cell data and restore the structure of the data. It also provides data preprocessing functionality such as dimensionality reduction and gene expression visualization.

* Palantir

#### Installation and dependencies
1. SCAnalysis has been implemented in Python3 and can be installed using:

        $> git clone git://github.com/helenjin/scanalysis.git
        $> cd scanalysis
        $> sudo -H pip3 install .

2. SCAnalysis depends on a number of `python3` packages available on pypi and these dependencies are listed in `setup.py`.
All the dependencies will be automatically installed using the above commands.

3. To uninstall:
		
		$> sudo -H pip3 uninstall scanalysis

#### Usage
##### Interactive command line
A tutorial on SCAnalysis usage and results visualization for single cell RNA-seq data can be found in this notebook: 
https://nbviewer.jupyter.org/github/helenjin/scanalysis/blob/master/notebooks/SCAnalysis.ipynb

##### GUI
Unfortunately, a python GUI is currently not available for SCAnalysis. Updates to follow.
However, it would be invoked, like so:

		$> sca_gui.py

#### Citations
Setty M, Tadmor MD, Reich-Zeliger S, Angel O, Salame TM, Kathail P, Choi K, Bendall S, Friedman N, Pe’er D. "Wishbone identifies bifurcating developmental trajectories from single-cell data." Nat. Biotech. 2016 April 12. <http://dx.doi.org/10.1038/nbt.3569>

van Dijk, David, et al. "MAGIC: A diffusion-based imputation method reveals gene-gene interactions in single-cell RNA-sequencing data." BioRxiv (2017): 111591. <http://www.biorxiv.org/content/early/2017/02/25/111591>

##### Original Source Code
Wishbone: <http://github.com/ManuSetty/wishbone.git>

MAGIC: <http://github.com/pkathail/magic.git>