

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
        "z_random_state": 42,
        "z_mlflow_server": "local",
    }

    prod_settings = {
        "z_rff_components": [4000], #[1000, 2000, 4000],
        "z_gamma" : [(2**i) for i in range(-6,6)], #-9,8)]
        "z_num_eigs": [100, 200, 400, 1000],
        "z_eig_epochs": [50, 100, 150]
    }

    m, best_params = hyperparameter_search("dmkde_adp", database, parent_path, prod_settings, settings)

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

#databases = ["cover"]

if start == 0:
    process_type = '/device:GPU:0'
elif start == 1:
    process_type = '/device:GPU:1'
elif start == 2:
    process_type = '/device:CPU:0'
else:
    process_type = '/device:GPU:0'

with tf.device(process_type):
    for database in databases:
        execution(database)


