try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("qaddemadac", "/home/jagallegom/")

from run_experiment_hyperparameter_search import hyperparameter_search
from experiment import experiment

import tensorflow as tf

databases = ["arrhythmia", "glass", "ionosphere", "letter", "mnist", "musk", "optdigits",
             "pendigits", "pima", "satellite", "satimage-2", "spambase", "vertebral", "vowels", "wbc",
             "breastw", "wine", "cardio", "speech", "thyroid", "annthyroid", "mammography", "shuttle", "cover"]

with tf.device('/device:GPU:1'):
    for database in databases:

        settings = {
            "z_prefix": "NEWW_",
            "z_dataset": database,
            "z_batch_size": 256,
            "z_threshold": 0.0,
            "z_epochs": 128,
            "z_base_lr": 1e-3,
            "z_end_lr": 1e-7,
            "z_power": 1,
            "z_decay_steps": int(100),
            "z_autoencoder_epochs": int(128),
            "z_autoencoder_batch_size": int(256),
            "z_adaptive_base_lr": 1e-3,
            "z_adaptive_end_lr": 1e-5,
            "z_adaptive_decay_steps": 100,
            "z_adaptive_power": 1,
            "z_adaptive_batch_size": 256,
            "z_adaptive_epochs": 24,
            "z_adaptive_random_state": 42,
            "z_adaptive_num_of_samples": None,
            "z_random_search": False,
            "z_random_search_random_state": 402,
            "z_random_search_iter": 0,
            "z_select_regularizer": None,
            "z_verbose": 0,
        }

        prod_settings = {
            "z_adaptive_fourier_features_enable": ['True'],
            "z_sigma": [2**i for i in range(-7,8)],
            "z_rff_components": [500,1000,2000],
            "z_max_num_eigs": [0.05,0.1,0.2,0.5,1],
            "z_sequential": ["(64,20,10,4)","(128,64,32,8)","(128,32,2)","(64,32,16)"],
            "z_alpha": [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1],
            "z_enable_reconstruction_metrics": ['True', 'False'],
            #"z_layer" : [tf.keras.layers.LeakyReLU(),tf.keras.layers.tanh()]
            "z_layer_name" : ["tanh"]
        }

        m, best_params = hyperparameter_search("leand", database, parent_path, prod_settings, settings)
        
        experiment(best_params, m, best=True)
