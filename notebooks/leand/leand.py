try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("LEAND", "/home/oabustosb/Desktop/")

from run_experiment_hyperparameter_search import hyperparameter_search
from experiment import experiment


databases = ["arrhythmia", "glass", "ionosphere", "letter", "mnist", "musk", "optdigits",
             "pendigits", "pima", "satellite", "satimage-2", "vertebral", "vowels", "wbc",
             "breastw", "wine", "cardio", "speech", "thyroid", "annthyroid", "mammography", "shuttle", "cover"]


for database in databases:

    settings = {
        "z_dataset": database,
        "z_batch_size": 256,
        "z_threshold": 0.0,
        "z_epochs": 150,
        "z_base_lr": 1e-2,
        "z_end_lr": 1e-8,
        "z_power": 1,
        "z_decay_steps": int(200),
        "z_autoencoder_epochs": int(200),
        "z_autoencoder_batch_size": int(256),
        "z_adaptive_base_lr": 1e-2,
        "z_adaptive_end_lr": 1e-5,
        "z_adaptive_decay_steps": 100,
        "z_adaptive_power": 1,
        "z_adaptive_batch_size": 256,
        "z_adaptive_epochs": 1, #50,
        "z_adaptive_random_state": 42,
        "z_random_search": True,
        "z_random_search_random_state": 42,
        "z_random_search_iter": 200,
    }

    prod_settings = {
        "z_adaptive_fourier_features_enable": [True],
        "z_sigma": [3], # [2**i for i in range(-7,8)],
        "z_rff_components": [1000], #[500,1000,2000],
        "z_max_num_eigs": [0.1], #[0.05,0.2,0.5,1],
        "z_sequential": ["(64,20,10,4)"], #"(128,64,32,8)","(128,32,2)","(64,32,16)"],
        "z_alpha": [1], #[0, 0.01, 0.1, 0.5, 0.9, 0.99, 1],
        "z_enable_reconstruction_metrics": [False, True]
    }

    m, best_params = hyperparameter_search("leand", database, parent_path, prod_settings, settings)
    
    experiment(best_params, m, best=True)
