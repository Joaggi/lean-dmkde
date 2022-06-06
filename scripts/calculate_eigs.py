import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np


def calculate_eigs(X_train, setting, i):

    fm_x = layers.QFeatureMapRFF(X_train.shape[1], dim=setting["z_rff_components"], 
                                 gamma=setting["z_gamma"], random_state=setting["z_random_state"])
    qmd = models.QMDensity(fm_x, setting["z_rff_components"])
    qmd.compile()
    qmd.fit(X_train, epochs=1, batch_size=setting["z_batch_size"], verbose=0)

    rho = qmd.weights[2]
    qmd2 = models.QMDensitySGD(X_train.shape[1], setting["z_rff_components"], num_eig=setting["z_max_num_eigs"], 
                               gamma=setting["z_gamma"], random_state=setting["z_random_state"])
    eig_vals = qmd2.set_rho(rho)
    opt_num_eigs = ( (np.asarray(eig_vals)) > 0.005 ).sum()

    if opt_num_eigs > 10:
        print(f"experiment_dmkde_sgd {i} num_eigs {opt_num_eigs}")    
        return (rho, opt_num_eigs)
    else:
        print(f"experiment_dmkde_sgd {i} num_eigs default {10}")    
        return (rho, 10)