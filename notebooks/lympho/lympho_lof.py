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
    "z_name_of_experiment": 'localoutlier-lympho1',
    "z_run_name": "localoutlier",
    "z_dataset": "lympho",
    "z_select_best_experiment": True
}

prod_settings = {"z_n_neighbors" : [2*i for i in range(1,26)], "z_nu": [i/100 for i in range(1,16)]}

params_int = ["z_n_neighbors"]
params_float = ["z_nu"]

mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

experiments(setting, prod_settings, params_int, params_float, mlflow)

