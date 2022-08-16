current_path = ""
import os

try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

#parent_path = initialization("leand", "/home/oabustosb/Desktop/")
parent_path = initialization("leand", "/Doctorado/")

import qmc.tf.layers as layers
import qmc.tf.models as models

import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from experiments import experiments

from mlflow_create_experiment import mlflow_create_experiment


setting = {
    "z_name_of_experiment": 'leand_v2_musk',
    "z_run_name": "leand",
    "z_dataset": "musk",
    "z_batch_size": 256,
    "z_select_best_experiment": True,
    "z_threshold": 0.0,
    "z_epochs": 200,
    "z_base_lr": 1e-2,
    "z_end_lr": 1e-9,
    "z_power": 1,
    "z_decay_steps": int(200),
    "z_autoencoder_epochs": int(200),
    "z_autoencoder_batch_size": int(256),
    "z_adaptive_base_lr": 1e-2,
    "z_adaptive_end_lr": 1e-5,
    "z_adaptive_decay_steps": 10,
    "z_adaptive_power": 1,
    "z_adaptive_batch_size": 256,
    "z_adaptive_epochs": 10,
    "z_random_search": True,
    "z_random_search_random_state": 42,
    "z_random_search_iter": 200,
    "z_layer": tf.keras.layers.LeakyReLU(), 
    "z_regularizer": tf.keras.regularizers.l1(10e-5),
}

prod_settings = { 
    "z_adaptive_fourier_features_enable": [True, False],
    "z_sigma": [2**i for i in range(-5,10)],
    "z_rff_components": [250,500,1000,2000],
    "z_max_num_eigs": [0.05,0.2,0.5,1],
    "z_sequential": [(64,32,16),(128,64,32,8),(128,32,2),(64,20,10,4)],
    "z_alpha": [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1], 
    "z_enable_reconstruction_metrics": [False, True]
}

prod_settings_rff = { 
    "z_adaptive_fourier_features_enable": [ False],
    "z_sigma": [4.0],
    "z_rff_components": [1000],
    "z_max_num_eigs": [0.2],
    "z_sequential": [(64,32,16)],
    "z_alpha": [0.99], 
    "z_enable_reconstruction_metrics": [ True]
}

prod_settings_aff = { 
    "z_adaptive_fourier_features_enable": [ True],
    "z_sigma": [2.0, 2.0],
    "z_rff_components": [2000],
    "z_max_num_eigs": [1.0],
    "z_sequential": [(64,20,10,4)],
    "z_alpha": [0.5], 
    "z_enable_reconstruction_metrics": [ True]
}
 
prod_settings = prod_settings_aff


mlflow = mlflow_create_experiment(setting["z_name_of_experiment"], server="server")

experiments(setting, prod_settings, mlflow)
