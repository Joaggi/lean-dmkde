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
    "z_name_of_experiment": 'isolation-thyroid',
    "z_run_name": "isolation",
    "z_dataset": "thyroid", 
    "z_select_best_experiment": True
}

prod_settings = {
    "z_n_estimators": [20*i for i in range(1,6)], 
    "z_max_samples" : [20*i for i in range(1,6)], 
    "z_nu": [i/50 for i in range(1,16)]
}

params_int = ["z_n_estimators", "z_max_samples"]
params_float = ["z_nu"]

mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

experiments(setting, prod_settings, params_int, params_float, mlflow)
