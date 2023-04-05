
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if path not in sys.path:
    sys.path.append(path)



try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("qaddemadac", path + "/")

import pandas as pd

import mlflow


from mlflow_get_experiment import mlflow_get_experiment

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
import os

algorithms = ["ablat_ae", "ablat_kde", "addmkde", "addmkde_sgd", "covariance", "isolation", 
              "lake", "leand", "lof", "oneclass", "pca-dmkde",
              "pyod-alad", "pyod-copod", "pyod-deepsvdd", "pyod-knn", "pyod-loda", "pyod-sogaal", "pyod-sos", 
              "pyod-vaebayes", "qadvaeff"]

databases = ["arrhythmia", "glass", "ionosphere", "letter", "mnist", "musk", "optdigits",
             "pendigits", "pima", "satellite", "satimage-2", "spambase", "vertebral", "vowels", "wbc",
             "breastw", "wine", "cardio", "speech", "thyroid", "annthyroid", "mammography", "shuttle", "cover"]

fig, axs = plt.subplots(9, 6, figsize=(80,80))

name_of_experiment =  'v9'

df = None

for j, dataset in enumerate(datasets):
    for i, algorithm in enumerate(algorithms):

        mlflow = mlflow_get_experiment(f"tracking.db", f"registry.db", name_of_experiment)

        client = MlflowClient()
        experiments_list = client.list_experiments()
        print(experiments_list)



        query = f"params.z_run_name = '{dataset}_{algorithm}' and params.z_step = 'test' and params.z_dimension = '{dimension}'"



        runs = mlflow.search_runs(experiment_ids="1", filter_string=query,
            run_view_type=ViewType.ACTIVE_ONLY, output_format="pandas")
        try:
            runs = runs.groupby(["params.z_algorithm", "params.z_dataset", "params.z_dimension"]).mean()

            if df is None:
                df = runs
            else:
                df = pd.concat([df, runs])
        except:
            pass

print(df.to_csv("conditional-density-estimation-gmm.csv"))
