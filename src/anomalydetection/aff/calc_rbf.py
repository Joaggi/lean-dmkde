import numpy as np 

def calc_rbf(dmrff, x1, x2):
    return dmrff.predict(np.concatenate([x1[:, np.newaxis, ...], 
                                         x2[:, np.newaxis, ...]], 
                                        axis=1),
                         batch_size=256)


