import GraphDiffusion
import pandas as pd

def run_diffusion_map(data, knn=10, normalization='smarkov', epsilon=1, n_diffusion_components=10):
        """ Run diffusion maps on the data. Run on the principal component projections
        for single cell RNA-seq data and on the expression matrix for mass cytometry data
        :param knn: Number of neighbors for graph construction to determine distances between cells
        :param epsilon: Gaussian standard deviation for converting distances to affinities
        :param n_diffusion_components: Number of diffusion components to Generalte
        :return: diffusion eigen vectors and eigen values, both as pandas DataFrames 
        """

        ## NEED to run PCA before running diffusion maps for single cell RNA-seq data ##
        
        # returns dictionary containing diffusion operator, weight matrix, diffusion eigen vectors, and diffusion eigen values
        d = GraphDiffusion.graph_diffusion.run_diffusion_map(data, knn, normalization, epsilon, n_diffusion_components)
        
        res1 = pd.DataFrame(d.get('EigenVectors'), index=data.index)      
        res2 = pd.DataFrame(d.get('EigenValues'))
        
        print("Successfully ran diffusion map, and returning EigenVectors and EigenValues")
        
        return res1, res2
