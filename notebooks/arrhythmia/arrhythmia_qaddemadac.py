current_path = ""
import os

try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("2021-3-anomaly-detection", "/Doctorado/")


import qmc.tf.layers as layers
import qmc.tf.models as models

import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from experiments import experiments

from mlflow_create_experiment import mlflow_create_experiment


setting = {
    "z_name_of_experiment": 'qaddemadac_arrhythmia',
    "z_run_name": "qaddemadac",
    "z_dataset": "arrhythmia",
    "z_rff_components": 1000,
    "z_batch_size": 128,
    "z_select_best_experiment": True,
    "z_threshold": 0.0,
    "z_epochs": 1000,
    "z_max_num_eigs": 100,
    "z_base_lr": 1e-3,
    "z_end_lr": 1e-7,
    "z_power": 1,
    "z_decay_steps": int(1000),
    "z_autoencoder_epochs": int(500),
    "z_autoencoder_batch_size": int(256),
    "z_adaptive_base_lr": 1e-1,
    "z_adaptive_end_lr": 1e-5,
    "z_adaptive_decay_steps": 1000,
    "z_adaptive_power": 1,
    "z_adaptive_batch_size": 512,
    "z_adaptive_epochs": 1000,
    "z_random_search": True,
    "z_random_search_random_state": 42,
    "z_random_search_iter": 100,
}

prod_settings = { 
    "z_adaptive_fourier_features_enable": [True, False],
    "z_sigma": [2**i for i in range(-20,20)],  
    "z_adaptive_input_dimension": [2,4,6,8,10]
}

params_int = ["z_rff_components", "z_batch_size", "z_epochs", "z_max_num_eigs"]
params_float = ["z_sigma", "z_threshold", "z_learning_rate"]
params_boolean = ["z_adaptive_fourier_features_enable"]


mlflow = mlflow_create_experiment(setting["z_name_of_experiment"], server="server")

experiments(setting, prod_settings, params_int, params_float, mlflow, params_boolean)
