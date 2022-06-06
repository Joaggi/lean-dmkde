import pytest 
from neuraldensityestimation.load_potential_1 import load_potential_1 
import matplotlib.pylab as plt

def test_load_potential_1():
    X_train, X_train_density, X_test, X_test_density = load_potential_1(1000, 1000, 2)
    plt.axes(frameon = 0)
    plt.grid()
    plt.scatter(X_test[:,0],  X_test[:,1], c = X_test_density , alpha = .5, s = 3, linewidths= 0.0000001)
    plt.colorbar()
    plt.title('potential_1 dataset')
    plt.savefig('reports/potential_1_dataset.png',dpi = 300)
    plt.show()
