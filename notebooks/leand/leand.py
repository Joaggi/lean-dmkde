
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
            "z_dataset_random_state": 42,
            "z_batch_size": 256,
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
            "z_random_search_random_state": 402,
            "z_random_search_iter": 50,
            "z_log_loss": False,
            "z_verbose": 1,
            "z_mlflow_server": "local",

        }

        prod_settings = {
            "z_adaptive_fourier_features_enable": ['False', 'True'],
            "z_adaptive_random_samples_enable": ["True", "False"],
            "z_enable_reconstruction_metrics": ['True', 'False'],
            "z_percentile" : [ 0.2,0.3, 0.5, 0.8],
            "z_multiplier" : [0.7, 1, 1.3],
            "z_rff_components": [16, 32, 64, 128, 256, 512,1024,2000],
            "z_max_num_eigs": [0.05,0.1,0.2,0.5,1],
            "z_sequential": ["(64,20,10,4)","(128,64,32,8)","(128,32,2)","(64,32,16)", "(256,128,32,4)", "(128, 64, 8)", "(256,)", "(128,)", "(64,)", "(8,)", "(64,16)", "(32,8)"],
            #"z_sequential": ["(256,)"],
            #"z_sequential": ["(128, 256, 512, 1024)", "(64,128,256)", "(32,64,256)"],
            "z_alpha": [0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1],
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

        #prod_settings_arrhythmia={
        #    "z_adaptive_fourier_features_enable": ['True'],
        #    "z_adaptive_random_samples_enable": ["True"],
        #    "z_enable_reconstruction_metrics": ['True'],
        #    "z_sigma": [0.50],
        #    "z_adaptive_epochs": [64],
        #    "z_epochs": [512],
        #    "z_rff_components": [1000,1000],
        #    "z_max_num_eigs": [0.5],
        #    "z_sequential": ["(128,32,2)"],
        #    "z_alpha": [0.1],
        #    "z_base_lr" : [1e-3],
        #    "z_adaptive_base_lr" : [1e-3],
        #    "z_layer_name" : ["tanh"],
        #    "z_activity_regularizer_value": [0.01],
        #    "z_autoencoder_type": ["unconstrained"],
        #    "z_activity_regularizer": [ None],
        #    "z_kernel_regularizer": [ "None"],
        #    "z_kernel_contraint": ["None"],
        #} 




        if database == "annthyroid":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [0.03125, 0.04, 0.03, 0.025, 0.045],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [2000],
                "z_max_num_eigs": [0.2],
                "z_sequential": ["(256,128,32)"],
                "z_alpha": [0.1, 0.05, 0.15 ],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": ["l2"],
                "z_activity_regularizer_value": [0.001],
                "z_autoencoder_type": ["tied" ],
                "z_kernel_regularizer": ["None"],
                "z_kernel_contraint": ["None"],
            }



        if database == "arrhythmia":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [0.26, 0.258, 0.262],
                "z_adaptive_epochs": [128],
                "z_epochs": [512],
                "z_rff_components": [32],
                "z_max_num_eigs": [1],
                "z_sequential": ["(128,)"],
                "z_alpha": [0.1],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": [ "l1"],
                "z_activity_regularizer_value": [0.01],
                "z_autoencoder_type": ["tied"],
                "z_kernel_regularizer": [ "weights_orthogonality"],
                "z_kernel_contraint": ["None"],
            }


        if database == "breastw":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [9, 9.42, 10, 9.2, 9.7],
                "z_adaptive_epochs": [128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [128],
                "z_max_num_eigs": [0.05],
                "z_sequential": ["(256,)"],
                "z_alpha": [0.99],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": [ "None"],
                "z_activity_regularizer_value": [None],
                "z_autoencoder_type": ["unconstrained"],
                "z_kernel_regularizer": [ "None"],
                "z_kernel_contraint": ["None"],
            }

        if database == "breastw":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [0.0625, 0.05,0.07,0.06],
                "z_adaptive_epochs": [128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [1000],
                "z_max_num_eigs": [0.5],
                "z_sequential": ["(256,)"],
                "z_alpha": [0.999],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": [ "None"],
                "z_activity_regularizer_value": [None],
                "z_autoencoder_type": ["unconstrained"],
                "z_kernel_regularizer": [ "None"],
                "z_kernel_contraint": ["None"],
            }



        if database == "cardio":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [0.015625,0.02,0.01,0.016,0.014],
                "z_adaptive_epochs": [128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [1000],
                "z_max_num_eigs": [0.5],
                "z_sequential": ["(64,20,10,4)"],
                "z_alpha": [0.99],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": [ "None"],
                "z_activity_regularizer_value": [None],
                "z_autoencoder_type": ["unconstrained"],
                "z_kernel_regularizer": [ "None"],
                "z_kernel_contraint": ["None"],
            }


        if database == "cover":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [0.5, 0.45, 0.55],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [1000],
                "z_max_num_eigs": [0.5],
                "z_sequential": ["(8,)"],
                "z_alpha": [1],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": [ "l2"],
                "z_activity_regularizer_value": [0.1],
                "z_autoencoder_type": ["tied"],
                "z_kernel_regularizer": [ "None"],
                "z_kernel_contraint": ["unit_norm"],
            }

        if database == "glass":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [0.5, 0.4, 0.55, 0.45],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [1000, 512],
                "z_max_num_eigs": [0.5, 0.45, 0.55],
                "z_sequential": ["(128,32,2)"],
                "z_alpha": [0.1],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": [ "None"],
                "z_activity_regularizer_value": [None],
                "z_autoencoder_type": ["unconstrained"],
                "z_kernel_regularizer": [ "None"],
                "z_kernel_contraint": ["None"],
            }

        if database == "ionosphere":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [1.5, 2, 2.5, 1.9, 2.1],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [2000],
                "z_max_num_eigs": [0.05, 0.1, 0.04, 0.06],
                "z_sequential": ["(64)"],
                "z_alpha": [0.999],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["LeakyReLU"],
                "z_activity_regularizer": [ "l2"],
                "z_activity_regularizer_value": [0.01],
                "z_autoencoder_type": ["unconstrained"],
                "z_kernel_regularizer": [ "None"],
                "z_kernel_contraint": ["None"],
            }

        if database == "letter":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [1.9, 2, 2.1, 1.8, 2.2],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [2000],
                "z_max_num_eigs": [0.04, 0.05, 0.06],
                "z_sequential": ["(64)"],
                "z_alpha": [0.999],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["LeakyReLU"],
                "z_activity_regularizer": [ "l2"],
                "z_activity_regularizer_value": [0.01],
                "z_autoencoder_type": ["unconstrained"],
                "z_kernel_regularizer": [ "None"],
                "z_kernel_contraint": ["None"],
            }


        if database == "mammography":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [128, 120, 135],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [2000],
                "z_max_num_eigs": [1, 0.5, 0.9],
                "z_sequential": ["(64,32,16)"],
                "z_alpha": [0.1],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": [ "l2"],
                "z_activity_regularizer_value": [0.0001],
                "z_autoencoder_type": ["tied"],
                "z_kernel_regularizer": [ "None"],
                "z_kernel_contraint": ["None"],
            }

        if database == "mnist":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [2.0, 1.9, 2.1, 1.5, 2.5, 1.99, 2.01],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [1000],
                "z_max_num_eigs": [0.5],
                "z_sequential": ["(256,)"],
                "z_alpha": [0.001],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": [ "None"],
                "z_activity_regularizer_value": [None],
                "z_autoencoder_type": ["unconstrained"],
                "z_kernel_regularizer": [ "None"],
                "z_kernel_contraint": ["None"],
            }


        if database == "musk":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [0.33246, 0.33, 0.34,0.33],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [64],
                "z_max_num_eigs": [1.0],
                "z_sequential": ["(128,)"],
                "z_alpha": [0.01, 0.1, 0.001],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": ["l1"],
                "z_activity_regularizer_value": [1.0],
                "z_autoencoder_type": ["tied"],
                "z_kernel_regularizer": [ "weights_orthogonality"],
                "z_kernel_contraint": ["None"],
            }


        if database == "optdigits":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [0.03125, 0.02, 0.04],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [500],
                "z_max_num_eigs": [0.05],
                "z_sequential": ["(128,64,8)"],
                "z_alpha": [0.001, 0.01, 0.1],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["LeakyReLU"],
                "z_activity_regularizer": ["None"],
                "z_activity_regularizer_value": [None],
                "z_autoencoder_type": ["unconstrained" ],
                "z_kernel_regularizer": ["None"],
                "z_kernel_contraint": ["None"],
            }

        if database == "pendigits":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [0.5, 0.4, 0.6],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [1000],
                "z_max_num_eigs": [0.5],
                "z_sequential": ["(128,32,2)"],
                "z_alpha": [0.1, 0.05, 0.15],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": ["None"],
                "z_activity_regularizer_value": [None],
                "z_autoencoder_type": ["unconstrained" ],
                "z_kernel_regularizer": ["None"],
                "z_kernel_contraint": ["None"],
            }


        if database == "pima":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [1, 1.2, 1.4],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [2000],
                "z_max_num_eigs": [1.0],
                "z_sequential": ["(128,64,32,8)"],
                "z_alpha": [1.0, 0.99, 0.9],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": ["uncorrelated_features"],
                "z_activity_regularizer_value": [0.001],
                "z_autoencoder_type": ["tied" ],
                "z_kernel_regularizer": ["None"],
                "z_kernel_contraint": ["unit_norm"],
            }


        if database == "satellite":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [16.0, 15.0, 17.0],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [2000],
                "z_max_num_eigs": [0.05],
                "z_sequential": ["(64,16)"],
                "z_alpha": [0.99, 0.9, 0.999],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": ["None"],
                "z_activity_regularizer_value": [None],
                "z_autoencoder_type": ["unconstrained" ],
                "z_kernel_regularizer": ["None"],
                "z_kernel_contraint": ["None"],
            }



        if database == "satimage-2":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [0.57, 0.6, 0.55, 0.575],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [1024],
                "z_max_num_eigs": [0.2],
                "z_sequential": ["(128,)"],
                "z_alpha": [0.001],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": ["l1"],
                "z_activity_regularizer_value": [0.001 ],
                "z_autoencoder_type": ["tied" ],
                "z_kernel_regularizer": ["None"],
                "z_kernel_contraint": ["unit_norm"],
            }


        if database == "shuttle":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [2.0, 1.5, 2.5, 1.7, 2.3],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [1000],
                "z_max_num_eigs": [0.5],
                "z_sequential": ["(256,)"],
                "z_alpha": [0.001],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": ["uncorrelated_features"],
                "z_activity_regularizer_value": [10.0 ],
                "z_autoencoder_type": ["tied" ],
                "z_kernel_regularizer": ["None"],
                "z_kernel_contraint": ["None"],
            }

        if database == "spambase":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [32.0, 31, 27, 35, 37],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [1000],
                "z_max_num_eigs": [0.05],
                "z_sequential": ["(256,)"],
                "z_alpha": [0.0, 0.05, 0.001],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": ["l1"],
                "z_activity_regularizer_value": [0.1],
                "z_autoencoder_type": ["unconstrained" ],
                "z_kernel_regularizer": ["None"],
                "z_kernel_contraint": ["None"],
            }

        if database == "speech":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [0.03125, 0.02, 0.04, 0.05],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [2000],
                "z_max_num_eigs": [0.2],
                "z_sequential": ["(128,)"],
                "z_alpha": [0.01, 0.02,0.005],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": ["uncorrelated_features"],
                "z_activity_regularizer_value": [1.0],
                "z_autoencoder_type": ["unconstrained" ],
                "z_kernel_regularizer": ["None"],
                "z_kernel_contraint": ["unit_norm"],
            }


        if database == "thyroid":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [1.0, 0.8, 1.2, 0.98, 1.02],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [500],
                "z_max_num_eigs": [0.5],
                "z_sequential": ["(256,)"],
                "z_alpha": [0.01,0.02,0.005],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": ["None"],
                "z_activity_regularizer_value": [None],
                "z_autoencoder_type": ["tied" ],
                "z_kernel_regularizer": ["None"],
                "z_kernel_contraint": ["None"],
            }


        if database == "vertebral":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [1.0, 1.1, 0.9, 0.5, 1.5],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [500],
                "z_max_num_eigs": [0.05],
                "z_sequential": ["(128,64,8)"],
                "z_alpha": [0.99,0.98,0.999,0.95],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["LeakyReLU"],
                "z_activity_regularizer": ["uncorrelated_features"],
                "z_activity_regularizer_value": [0.01],
                "z_autoencoder_type": ["tied" ],
                "z_kernel_regularizer": ["weights_orthogonality"],
                "z_kernel_contraint": ["None"],
            }



        if database == "vowels":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [2.0, 1.5, 2.5, 1.9, 2.1],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [2000],
                "z_max_num_eigs": [0.05],
                "z_sequential": ["(64,)"],
                "z_alpha": [0.999,0.99,0.9],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["LeakyReLU"],
                "z_activity_regularizer": ["l2"],
                "z_activity_regularizer_value": [0.01],
                "z_autoencoder_type": ["tied" ],
                "z_kernel_regularizer": ["None"],
                "z_kernel_contraint": ["None"],
            }


        if database == "wbc":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [7.74, 7.5, 8],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [256],
                "z_max_num_eigs": [0.05],
                "z_sequential": ["(64)"],
                "z_alpha": [0.99,0.999,0.9],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": ["uncorrelated_features"],
                "z_activity_regularizer_value": [1.0],
                "z_autoencoder_type": ["unconstrained" ],
                "z_kernel_regularizer": ["None"],
                "z_kernel_contraint": ["None"],
            }


        if database == "wine":
            prod_settings={
                "z_adaptive_fourier_features_enable": ['True'],
                "z_adaptive_random_samples_enable": ["True"],
                "z_enable_reconstruction_metrics": ['True'],
                "z_sigma": [8.0, 6, 7, 9, ,10, 8.5, 7.5],
                "z_adaptive_epochs": [32, 128, 256, 512],
                "z_epochs": [512],
                "z_rff_components": [500],
                "z_max_num_eigs": [0.2],
                "z_sequential": ["(256,128,32,4)"],
                "z_alpha": [0.999],
                "z_base_lr" : [1e-3],
                "z_adaptive_base_lr" : [1e-3],
                "z_layer_name" : ["tanh"],
                "z_activity_regularizer": ["l1"],
                "z_activity_regularizer_value": [1.0],
                "z_autoencoder_type": ["tied" ],
                "z_kernel_regularizer": ["weights_orthogonality"],
                "z_kernel_contraint": ["None"],
            }






        m, best_params = hyperparameter_search("leand", database, parent_path, prod_settings, settings)
        
        #experiment(best_params, m, best=False)

from run_experiment_hyperparameter_search import hyperparameter_search
from experiment import experiment

import tensorflow as tf
import sys

print(f"tf version: {tf.__version__}")


print(f"len sys.argv: {sys.argv}")

if len(sys.argv) >= 2 and sys.argv[1] != None:
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

databases = ["breastw"]

if start == 0:
    process_type = '/device:GPU:0'
elif start == 1:
    process_type = '/device:GPU:1'
elif start == 2:
    process_type = '/device:CPU:0'
else:
    process_type = '/device:GPU:0'

process_type = '/device:CPU:0'

print(f"process_type: {process_type}")
with tf.device(process_type):
    for database in databases:
        execution(database)


