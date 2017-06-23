from sklearn.decomposition import PCA

def run_pca(data, n_components=100, random=True):

    solver = 'randomized'
    if random != True:
        solver = 'full'

    pca = PCA(n_components=n_components, svd_solver=solver)
    return pca.fit_transform(data)
