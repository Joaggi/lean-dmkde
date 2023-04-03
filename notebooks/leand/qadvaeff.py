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



def execution(database):
        settings = {
            "z_prefix": "v8-only-autoencoder-",
            "z_dataset": database,
            "z_batch_size": 256,
            "z_threshold": 0.0,
            "z_epochs": 256,
            "z_base_lr": 1e-3,
            "z_end_lr": 1e-7,
            "z_power": 1,
            "z_decay_steps": int(100),
            "z_autoencoder_epochs": int(128),
            "z_autoencoder_batch_size": int(256),
            "z_autoencoder_is_trainable": 'True',
            "z_autoencoder_is_alone_optimized": 'False',
            "z_adaptive_base_lr": 1e-3,
            "z_adaptive_end_lr": 1e-5,
            "z_adaptive_decay_steps": 16,
            "z_adaptive_power": 1,
            "z_adaptive_batch_size": 256,
            "z_adaptive_epochs": 16,
            "z_adaptive_random_state": None,
            "z_adaptive_num_of_samples": 10000,
            "z_random_search": True,
            "z_random_search_random_state": None,
            "z_random_search_iter": 50,
            "z_verbose": 1,
            "z_mlflow_server": "local",

        }

        prod_settings = {
            "z_adaptive_fourier_features_enable": ['False', 'True'],
            #"z_adaptive_fourier_features_enable": ['True'],
            #"z_adaptive_fourier_features_enable": ['False'],
            "z_adaptive_random_samples_enable": ["True", "False"],
            "z_sigma": [2**i for i in range(-7,8)],
            "z_rff_components": [500,1000,2000],
            "z_max_num_eigs": [0.05,0.1,0.2,0.5,1],
            "z_sequential": ["(64,20,10,4)","(128,64,32,8)","(128,32,2)","(64,32,16)", "(256,128,32,4)", "(128, 64, 8)", "(256,)", "(128,)", "(64,)", "(8,)", "(64,16)"],
            #"z_sequential": ["(256,)"],
            #"z_sequential": ["(128, 256, 512, 1024)", "(64,128,256)", "(32,64,256)"],
            "z_alpha": [0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1],
            "z_enable_reconstruction_metrics": ['True', 'False'],
            #"z_layer" : [tf.keras.layers.LeakyReLU(),tf.keras.layers.tanh()]
            "z_base_lr" : [1e-2, 1e-3],
            "z_adaptive_base_lr" : [1e-2, 1e-3],
            "z_layer_name" : ["tanh", "LeakyReLU"],
            "z_activity_regularizer_value": [0.001, 0.01, 0.1,1, 10],
            "z_autoencoder_type": ["unconstrained", "tied"],
            "z_activity_regularizer": ["uncorrelated_features", "l1","l2", None],
            "z_kernel_regularizer": ["weights_orthogonality", "None"],
            "z_kernel_contraint": ["unit_norm", "None"],
        }

        m, best_params = hyperparameter_search("qadvaeff", database, parent_path, prod_settings, settings)
        
        #experiment(best_params, m, best=False)

from run_experiment_hyperparameter_search import hyperparameter_search
from experiment import experiment

import tensorflow as tf
import sys

print(f"tf version: {tf.__version__}")


print(sys.argv)

if len(sys.argv) > 1 and sys.argv[1] != None:
    start = int(sys.argv[1])
    jump = 3

else:
    start = 0
    jump = 1

    


databases = ["arrhythmia", "glass", "ionosphere", "letter", "mnist", "musk", "optdigits",
             "pendigits", "pima", "satellite", "satimage-2", "spambase", "vertebral", "vowels", "wbc",
             "breastw", "wine", "cardio", "speech", "thyroid", "annthyroid", "mammography", "shuttle", "cover"]

databases = databases[start::jump]
print(databases)

#databases = ["cover"]

if start == 0:
    process_type = '/device:GPU:0'
elif start == 1:
    process_type = '/device:GPU:1'
elif start == 2:
    process_type = '/device:CPU:0'
else:
    process_type = '/device:GPU:1'

with tf.device(process_type):
    for database in databases:
        execution(database)




