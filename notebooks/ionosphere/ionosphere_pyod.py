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
    sys.path.append('submodules/pyod/')
    print(sys.path)
else:
    import sys
    sys.path.append('submodules/qmc/')
    sys.path.append('submodules/pyod/')
    sys.path.append('data/')
    print(sys.path)


print(os.getcwd())

sys.path.append('scripts/')


from mlflow_create_experiment import mlflow_create_experiment

import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from experiments import experiments

algorithms = ["knn", "sos", "copod", "loda", "vae", "deepsvdd"]

for algorithm in algorithms:

    setting = {
        "z_name_of_experiment": ("pyod-"+algorithm+'-ionosphere'),
        "z_run_name": ("pyod-"+algorithm),
        "z_dataset": "ionosphere",
        "z_select_best_experiment": True
    }    

    if algorithm == "knn":
        prod_settings = {"z_nu": [i/50 for i in range(5,23)], "z_n_neighbors" : [10*i for i in range(1,11)]}
        params_int = ["z_n_neighbors"]
        params_float = ["z_nu"]
    elif algorithm == "sos":
        prod_settings = {"z_nu": [i/50 for i in range(5,23)], "z_perplexity" : [10.0*i for i in range(1,11)]}
        setting.update(z_tol = 1e-5)
        params_int = []
        params_float = ["z_perplexity", "z_nu", "z_tol"]
    elif algorithm == "copod":
        prod_settings = {"z_nu": [i/50 for i in range(5,23)]}
        params_int = []
        params_float = ["z_nu"]
    elif algorithm == "loda":
        prod_settings = {"z_nu": [i/100 for i in range(10,46)]}
        params_int = []
        params_float = ["z_nu"]
    elif algorithm == "vae":
        prod_settings = {"z_nu": [i/50 for i in range(5,23)]}
        setting.update(z_batch_size = 16, z_epochs = 80)
        params_int = ["z_batch_size", "z_epochs"]
        params_float = ["z_nu"] 
    elif algorithm == "deepsvdd":
        prod_settings = {"z_nu": [i/100 for i in range(10,46)]}
        setting.update(z_epochs = 0)
        params_int = ["z_epochs"]
        params_float = ["z_nu"]   
    else:
        prod_settings = {"z_nu": 0.30}
        params_int = []
        params_float = ["z_nu"]


    mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

    experiments(setting, prod_settings, params_int, params_float, mlflow)