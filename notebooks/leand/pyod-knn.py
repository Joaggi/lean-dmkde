
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
        "z_random_search_iter": 50,
        "z_verbose": 1,

        "z_mlflow_server": "local",
    }

    prod_settings = {
        "z_nu": nus,
        "z_n_neighbors": [10*i for i in range(1,11)],
    }

    if database == "arrhythmia":
        prod_settings = {
            "z_nu": [0.12],
            "z_n_neighbors": [80]
        }
    elif database == "glass":
        prod_settings = {
            "z_nu": [0.01],
            "z_n_neighbors": [10]
        }
    elif database == "ionosphere":
        prod_settings = {
            "z_nu": [0.40],
            "z_n_neighbors": [10]
        }
    elif database == "letter":
        prod_settings = {
            "z_nu": [0.03],
            "z_n_neighbors": [10]
        }
    elif database == "mnist":
        prod_settings = {
            "z_nu": [0.07],
            "z_n_neighbors": [50]
        }
    elif database == "musk":
        prod_settings = {
            "z_nu": [0.03],
            "z_n_neighbors": [50]
        }
    elif database == "optdigits":
        prod_settings = {
            "z_nu": [0.01],
            "z_n_neighbors": [40]
        }
    elif database == "pendigits":
        prod_settings = {
            "z_nu": [0.04],
            "z_n_neighbors": [80]
        }
    elif database == "pima":
        prod_settings = {
            "z_nu": [0.36],
            "z_n_neighbors": [60]
        }
    elif database == "satellite":
        prod_settings = {
            "z_nu": [0.29],
            "z_n_neighbors": [100]
        }
    elif database == "satimage-2":
        prod_settings = {
            "z_nu": [0.01],
            "z_n_neighbors": [90]
        }
    elif database == "spambase":
        prod_settings = {
            "z_nu": [0.22],
            "z_n_neighbors": [10]
        }
    elif database == "vertebral":
        prod_settings = {
            "z_nu": [0.12],
            "z_n_neighbors": [10]
        }
    elif database == "vowels":
        prod_settings = {
            "z_nu": [0.02],
            "z_n_neighbors": [10]
        }
    elif database == "wbc":
        prod_settings = {
            "z_nu": [0.08],
            "z_n_neighbors": [50]
        }
    elif database == "breastw":
        prod_settings = {
            "z_nu": [0.36],
            "z_n_neighbors": [50]
        }
    elif database == "wine":
        prod_settings = {
            "z_nu": [0.13],
            "z_n_neighbors": [10]
        }
    elif database == "cardio":
        prod_settings = {
            "z_nu": [0.13],
            "z_n_neighbors": [100]
        }
    elif database == "speech":
        prod_settings = {
            "z_nu": [0.02],
            "z_n_neighbors": [30]
        }
    elif database == "thyroid":
        prod_settings = {
            "z_nu": [0.02],
            "z_n_neighbors": [40]
        }
    elif database == "annthyroid":
        prod_settings = {
            "z_nu": [0.04],
            "z_n_neighbors": [60]
        }
    elif database == "mammography":
        prod_settings = {
            "z_nu": [0.02],
            "z_n_neighbors": [40]
        }
    elif database == "shuttle":
        prod_settings = {
            "z_nu": [0.01],
            "z_n_neighbors": [50]
        }
    elif database == "cover":
        prod_settings = {
            "z_nu": [0.01],
            "z_n_neighbors": [90]
        }



    m, best_params = hyperparameter_search("pyod-knn", database, parent_path, prod_settings, settings)

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


