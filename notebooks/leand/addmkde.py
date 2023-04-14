
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
        "z_experiment": "v10",
        "z_dataset": database,
        "z_num_samples": 10000,
        "z_batch_size": 32,
        "z_threshold": 0.0,
        "z_mlflow_server": "local",
        "z_adaptive_base_lr": 1e-3,
        "z_adaptive_end_lr": 1e-5,
        "z_adaptive_decay_steps": 16,
        "z_adaptive_power": 1,
        "z_adaptive_batch_size": 256,
        "z_adaptive_epochs": 32,
        "z_adaptive_random_state": None,
        "z_adaptive_num_of_samples": 10000,
        "z_random_search": True,
        "z_random_search_random_state": 402,
        "z_random_search_iter": 1,
        "z_verbose": 1,

    }

    prod_settings = {
        "z_rff_components": [16, 32, 64, 128, 256, 512,1024, 2048, 4000],
        #"z_gamma" : [2**i for i in range(-9,8)],
        "z_percentile" : [0.05, 0.1, 0.2,0.3, 0.5, 0.8, 1],
        "z_multiplier" : [0.2, 0.5, 1, 1.5, 2],
        "z_adaptive_fourier_features_enable": ['False', 'True'],
        "z_adaptive_random_samples_enable": ["True", "False"],
    }

    m, best_params = hyperparameter_search("addmkde", database, parent_path, prod_settings, settings)

    experiment(best_params, m, best=True)


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

#databases = databases[start::jump]
print(databases)

#databases = ["arrhythmia"]

if start == 0:
    process_type = '/device:GPU:0'
elif start == 1:
    process_type = '/device:GPU:1'
elif start == 2:
    process_type = '/device:CPU:0'
else:
    process_type = '/device:GPU:0'


process_type = '/device:CPU:0'

with tf.device(process_type):
    for database in databases:
        execution(database)


