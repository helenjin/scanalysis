from sklearn.decomposition import PCA
import pandas as pd

def run_pca(data, n_components=100, random=True):
    """
    :param: data (normalized dataframe object)
    
    :return: pca_projections (DataFrame object), pca.explained_variance_ratio_ (ndarray object)
    """
    solver = 'randomized'
    if random != True:
        solver = 'full'

    pca = PCA(n_components=n_components, svd_solver=solver)
    
    pca_projections = pca.fit_transform(data)
    
    pca_projections = pd.DataFrame(pca_projections)
    
    pca_projections.index = data.index.tolist()
    
    print("Successfully ran PCA, and returning:\npca_projections\npca.explained_variance_ratio_") 
    return pca_projections, pca.explained_variance_ratio_
