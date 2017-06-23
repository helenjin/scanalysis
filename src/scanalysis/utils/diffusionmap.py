import GraphDiffusion

def run_diffusion_map(data, knn=10, normalization='smarkov', epsilon=1, n_diffusion_components=10):
        GraphDiffusion.graph_diffusion.run_diffusion_map(data, knn, normalization, epsilon, n_diffusion_components)

