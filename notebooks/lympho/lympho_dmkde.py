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
    print(sys.path)
else:
    import sys
    sys.path.append('submodules/qmc/')
    sys.path.append('data/')
    print(sys.path)


print(os.getcwd())

sys.path.append('scripts/')


from mlflow_create_experiment import mlflow_create_experiment

import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from experiments import experiments

setting = {
    "z_name_of_experiment": 'dmkde-lympho',
    "z_run_name": "dmkde",
    "z_dataset": "lympho", 
    "z_rff_components": 1000,
    "z_batch_size": 8,
    "z_select_best_experiment": True,
    "z_threshold": 0.0
}

prod_settings = {"z_gamma": [2**i for i in range(-10,5)]}
#prod_settings = {"z_gamma" : [2**-6]}

params_int = ["z_rff_components", "z_batch_size"]
params_float = ["z_gamma", "z_threshold"]


mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

experiments(setting, prod_settings, params_int, params_float, mlflow)
