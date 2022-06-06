current_path = ""


try:  
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    import os
    import sys 
    sys.path.append('submodules/qmc/') 
    #sys.path.append('../../../../submodules/qmc/')
    print(sys.path)
else:
    import sys
    sys.path.append('submodules/qmc/')
    sys.path.append('data/')
    #sys.path.append('../../../../submodules/qmc/')
    print(sys.path)
    # %cd ../../

print(os.getcwd())

sys.path.append('scripts/')


import qmc.tf.layers as layers
import qmc.tf.models as models

import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from experiments import experiments

from mlflow_create_experiment import mlflow_create_experiment


setting = {
    "z_name_of_experiment": 'dmkde_adp-ionosphere',
    "z_run_name": "dmkde_adp",
    "z_dataset": "ionosphere",
    "z_rff_components": 1000,
    "z_num_samples": 10000,
    "z_batch_size": 16,
    "z_select_best_experiment": True,
    "z_threshold": 0.0
}

prod_settings = {"z_gamma": [2**i for i in range(-9,6)]}

params_int = ["z_rff_components", "z_batch_size", "z_num_samples"]
params_float = ["z_gamma", "z_threshold"]


mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

experiments(setting, prod_settings, params_int, params_float, mlflow)
