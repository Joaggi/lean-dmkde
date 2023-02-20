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
        "z_dataset": database,
        "z_epochs": 100,
        "z_batch_size": 100,
        "z_latent_dim": 2,
        "z_coder_layers": "[75,100]",
        "z_disc_xx": "[100,75]",
        "z_disc_zz": "[25,25]",
    }

    prod_settings = {
        "z_add_recon_loss": ['True','False'],
        "z_nu": nus,
    }

    m, best_params = hyperparameter_search("pyod-alad", database, parent_path, prod_settings, settings)
    
    experiment(best_params, m, best=True)

