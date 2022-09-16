current_path = ""
import os

try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("leand", "/Doctorado/")


import qmc.tf.layers as layers
import qmc.tf.models as models

import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from experiments import experiments

from mlflow_create_experiment import mlflow_create_experiment


setting = {
    "z_name_of_experiment": 'qadvaeff_cardio',
    "z_run_name": "qadvaeff",
    "z_dataset": "cardio",
    "z_rff_components": 1000,
    "z_batch_size": 128,
    "z_select_best_experiment": True,
    "z_threshold": 0.0,
    "z_epochs": 1000,
    "z_max_num_eigs": 100,
    "z_base_lr": 1e-2,
    "z_end_lr": 1e-4,
    "z_power": 1,
    "z_decay_steps": int(1000),
    "z_autoencoder_epochs": int(200),
    "z_autoencoder_batch_size": int(256),
    "z_adaptive_base_lr": 1e-2,
    "z_adaptive_end_lr": 1e-5,
    "z_adaptive_decay_steps": 30,
    "z_adaptive_power": 1,
    "z_adaptive_batch_size": 512,
    "z_adaptive_epochs": 40,
    "z_random_search": True,
    "z_random_search_random_state": 42,
    "z_random_search_iter": 200,
    "z_layer": tf.keras.layers.LeakyReLU(), 
    #"z_layer": tf.keras.activations.tanh, 
    "z_regularizer": tf.keras.regularizers.l1(10e-5),
}

prod_settings = { 
    "z_adaptive_fourier_features_enable": [False, True],
    "z_sigma": [0.001, 0.01, 0.2, 0.5, 1, 2, 8, 15, 100, 1000],
    "z_rff_components": [4000],
    "z_max_num_eigs": [0.1, 0.2, 0.5, 1.0],
    "z_sequential": [(22,20,15),(64,32,16),(128,64,32,8),(128,32,2),(64,20,10,4)],
    #"z_sequential": [(22,20,15)],
    "z_alpha": [0.1, 0.5, 0.9, 0.99, 1], 
    "z_enable_reconstruction_metrics": [False, True]
}


mlflow = mlflow_create_experiment(setting["z_name_of_experiment"], server="server")

experiments(setting, prod_settings, mlflow)
