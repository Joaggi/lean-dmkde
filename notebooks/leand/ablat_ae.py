try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("LEAND", "/home/oabustosb/Desktop/")

from run_experiment_hyperparameter_search import hyperparameter_search
from experiment import experiment


databases = ["arrhythmia", "glass", "ionosphere", "letter", "mnist", "musk", "optdigits",
             "pendigits", "pima", "satellite", "satimage-2", "spambase", "vertebral", "vowels", "wbc",
             "breastw", "wine", "cardio", "speech", "thyroid", "annthyroid", "mammography", "shuttle", "cover"]

for database in ["cover"]:

    settings = {
        "z_dataset": database,
        "z_threshold": 0.0,
        "z_base_lr": 1e-3,
        "z_end_lr": 1e-7,
        "z_power": 1,
        "z_decay_steps": int(1000),
        "z_autoencoder_epochs": int(320),
        "z_autoencoder_batch_size": int(256),
    }

    prod_settings = {
        "z_sequential": ['(128,64,32,8)','(128,32,2)','(64,20,10,4)'], #'(64,32,16)',
        "z_layername": ["relu", "sigmoid", "tanh"],
    }

    m, best_params = hyperparameter_search("autoencoder", database, parent_path, prod_settings, settings)

    experiment(best_params, m, best=True)
