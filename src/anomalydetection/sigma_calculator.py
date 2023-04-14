from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pylab as plt
import numpy as np

def sigma_calculator(X, percentile, multiplier):

    np.random.seed(seed = 100)
    idx_1 = np.random.randint(X.shape[0], size=1000)
    idx_2 = np.random.randint(X.shape[0], size=1000)
    distances_X = euclidean_distances(X[idx_1,:],X[idx_2,:])
    plt.axes(frameon = 0)
    plt.grid()
    plt.title('Histogram of distances')
    plt.hist(distances_X[np.triu_indices_from(distances_X, k=1)].ravel(), density = True, bins=40);

    sigma = np.percentile(distances_X, percentile)
    sigma = distances_X.min() if sigma == 0 else sigma
    sigma = sigma * multiplier 
     
    return sigma 
