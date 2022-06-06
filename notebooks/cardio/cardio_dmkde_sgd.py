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
    "z_name_of_experiment": 'dmkde_sgd_cardio',
    "z_run_name": "dmkde_sgd",
    "z_dataset": "cardio",
    "z_rff_components": 1000,
    "z_batch_size": 8,
    "z_select_best_experiment": True,
    "z_threshold": 0.0,
    "z_learning_rate": 1e-06,
    "z_epochs": 100,
    "z_max_num_eigs": 100
}

prod_settings = {"z_gamma": [2**i for i in range(-10,5)]}
#prod_settings = {"z_gamma" : [2**-6]}

params_int = ["z_rff_components", "z_batch_size", "z_epochs", "z_max_num_eigs"]
params_float = ["z_gamma", "z_threshold", "z_learning_rate"]


mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

experiments(setting, prod_settings, params_int, params_float, mlflow)
