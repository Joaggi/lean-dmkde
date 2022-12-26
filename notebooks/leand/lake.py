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


for database in databases:

    settings = {
        "z_dataset": database,
        "z_learning_rate": 1e-05,
        "z_iter_per_epoch": 1000,
    }

    prod_settings = {
        "z_batch_size": [100,200,500,1000],
        "z_enc_dec": [ '(45,35,30)','(20,15,15)','(60,25,20)' ]
    }

    m, best_params = hyperparameter_search("lake", database, parent_path, prod_settings, settings)
    
    experiment(best_params, m, best=True)
