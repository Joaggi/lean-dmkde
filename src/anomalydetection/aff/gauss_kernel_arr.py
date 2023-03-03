import numpy as np

def gauss_kernel_arr(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y, axis=1) ** 2)


