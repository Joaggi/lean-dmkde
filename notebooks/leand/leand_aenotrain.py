import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if path not in sys.path:
    sys.path.append(path)

try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("qaddemadac", path + "/")

from run_experiment_hyperparameter_search import hyperparameter_search
from experiment import experiment


databases = ["arrhythmia", "glass", "ionosphere", "letter", "mnist", "musk", "optdigits",
             "pendigits", "pima", "satellite", "satimage-2", "spambase", "vertebral", "vowels", "wbc",
             "breastw", "wine", "cardio", "speech", "thyroid", "annthyroid", "mammography", "shuttle", "cover"]


for database in databases[14:]:

    settings = {
        "z_experiment": "v10",
        "z_dataset": database,
        "z_batch_size": 128,
        "z_threshold": 0.0,
        "z_epochs": 60,
        "z_base_lr": 1e-2,
        "z_end_lr": 1e-7,
        "z_power": 1,
        "z_decay_steps": int(100),
        "z_autoencoder_epochs": int(100),
        "z_autoencoder_batch_size": int(256),
        "z_adaptive_base_lr": 1e-2,
        "z_adaptive_end_lr": 1e-5,
        "z_adaptive_decay_steps": 100,
        "z_adaptive_power": 1,
        "z_adaptive_batch_size": 256,
        "z_adaptive_epochs": 32,
        "z_adaptive_random_state": 42,
        "z_select_regularizer": 'l1',
        "z_mlflow_server": "local",
    }

    prod_settings = {
        "z_adaptive_fourier_features_enable": ['False'],
        "z_autoencoder_is_alone_optimized": ["True"],
        "z_autoencoder_is_trainable": ['False'],
        "z_sigma": [2**(i/2) for i in range(-7,8)], 
        "z_rff_components": [1000, 2000, 4000],
        "z_max_num_eigs": [1],
        "z_sequential": ["(40,20,10,4)","(64,32,16,8)","(32,8,2)","(128,64,32)"], #["(64,20,10,4)","(128,64,32,8)","(128,32,2)","(64,32,16)"],
        "z_alpha": [0.1)], 
        "z_enable_reconstruction_metrics": ['True'],
        "z_layer_name" : ["tanh"]
    }

    m, best_params = hyperparameter_search("leand_aenotrain", database, parent_path, prod_settings, settings)
    
    experiment(best_params, m, best=True)
