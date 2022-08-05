current_path = ""

import os


try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("qaddemadac", "/home/oabustosb/Desktop/")


import qmc.tf.layers as layers
import qmc.tf.models as models

import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from experiments import experiments

from mlflow_create_experiment import mlflow_create_experiment


setting = {
    "z_name_of_experiment": 'kde_shuttle',
    "z_run_name": "kde",
    "z_dataset": "shuttle",
    "z_select_best_experiment": True,
    "z_threshold": 0.0
}

prod_settings = {"z_kernel_fun": ['gaussian', 'exponential', 'linear', 'cosine', 'tophat', 'epanechnikov'],
                 "z_bandwidth" : [2**i for i in range(7,-8,-1)] }

params_int = []
params_float = ["z_threshold"]


mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

experiments(setting, prod_settings, params_int, params_float, mlflow)
