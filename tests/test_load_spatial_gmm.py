import pytest 
import sys
print(sys.path)

from neuraldensityestimation.load_spatial_gmm import load_spatial_gmm 
import matplotlib.pylab as plt

def test_load_spatial_gmm():
    X_train, X_train_density, _, _, X_train_label, _ = load_spatial_gmm(10000, 1000, 2)
    plt.axes(frameon = 0)
    plt.grid()
    plt.scatter(X_train[:,0],  X_train[:,1], c = X_train_density, alpha = .7, s = 4, linewidths= 0.0000001)
    plt.colorbar()
    plt.title('spatial_gmm dataset')
    plt.savefig('reports/spatial_gmm_dataset.png',dpi = 300)
    plt.show()
    
    plt.Figure()
    plt.axes(frameon = 0)
    plt.grid()
    plt.scatter(X_train[:,0],  X_train[:,1], c = X_train_label, alpha = .7, s = 10, linewidths= 0.0000001)
    plt.colorbar()
    plt.title('spatial_gmm dataset')
    plt.savefig('reports/spatial_gmm_dataset.png',dpi = 300)
    plt.show()
