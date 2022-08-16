current_path = ""
import os

try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("leand", "/home/oabustosb/Desktop/")


import qmc.tf.layers as layers
import qmc.tf.models as models

import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from experiments import experiments

from mlflow_create_experiment import mlflow_create_experiment


setting = {
    "z_name_of_experiment": 'autoencoder_pendigits',
    "z_run_name": "autoencoder",
    "z_dataset": "pendigits",
    "z_select_best_experiment": True,
    "z_threshold": 0.0,
    "z_base_lr": 1e-3,
    "z_end_lr": 1e-7,
    "z_power": 1,
    "z_decay_steps": int(1000),
    "z_autoencoder_epochs": int(320),
    "z_autoencoder_batch_size": int(256),
}

prod_settings = {
    #"z_alpha": [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1], 
    "z_sequential": [(64,32,16),(128,64,32,8),(128,32,2),(64,20,10,4)],
    "z_layer": ["relu", "sigmoid", "tanh"],
    "z_regularizer": ["l2", "l1", "l1_l2"]
}

params_int = ["z_power", "z_decay_steps", "z_autoencoder_epochs", "z_autoencoder_batch_size"]
params_float = ["z_sequential", "z_threshold", "z_base_lr", "z_end_lr"]
params_boolean = []


mlflow = mlflow_create_experiment(setting["z_name_of_experiment"], server="local")

experiments(setting, prod_settings, params_int, params_float, mlflow, params_boolean)
