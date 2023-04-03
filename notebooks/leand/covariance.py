

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

    nus = []
    if database=="spambase":
        nus = [(i/100) for i in range(16,25)]
    elif database in ["breastw","ionosphere","pima","satellite"]: 
        nus = [(i/100) for i in range(25,41)]
    elif database in ["speech","thyroid","mammography","cover","optdigits","pendigits","satimage-2"]:
        nus = [(i/100) for i in range(1,6)]
    else:
        nus = [(i/100) for i in range(1,16)]

    settings = {
        "z_experiment": "v10",
        "z_dataset": database,
        "z_random_search": True,
        "z_random_search_random_state": 402,
        "z_random_search_iter": 2,
        "z_verbose": 1,

        "z_mlflow_server": "local",
    }

    prod_settings = {
        "z_nu": nus
    }

    m, best_params = hyperparameter_search("covariance", database, parent_path, prod_settings, settings)
    
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


