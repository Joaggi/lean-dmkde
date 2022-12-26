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
        "z_threshold": 0.0
    }

    prod_settings = {
        "z_kernel_fun": ['gaussian', 'exponential', 'linear', 'cosine', 'tophat', 'epanechnikov'], 
        "z_bandwidth" : [2**i for i in range(7,-8,-1)]
    }

    m, best_params = hyperparameter_search("kde", database, parent_path, prod_settings, settings)

    experiment(best_params, m, best=True)
