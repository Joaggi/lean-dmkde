from pyod.models.knn import KNN
from pyod.models.sos import SOS
from pyod.models.copod import COPOD
from pyod.models.loda import LODA
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD

import numpy as np
from calculate_metrics import calculate_metrics


def experiment_pyod(X_train, y_train, X_test, y_test, settings, mlflow, best=False):

    for i, setting in enumerate(settings):
        #print(f"experiment_dmkdc {i} threshold {setting['z_threshold']}")
        with mlflow.start_run(run_name=setting["z_run_name"]):

            model = None

            if (setting["z_run_name"] == "pyod-knn"):
                model = KNN(contamination=setting["z_nu"], n_neighbors=setting["z_n_neighbors"])
                
            if (setting["z_run_name"] == "pyod-sos"):
                model = SOS(contamination=setting["z_nu"], perplexity=setting["z_perplexity"],
                            eps=setting["z_tol"])

            if (setting["z_run_name"] == "pyod-copod"):
                model = COPOD(contamination=setting["z_nu"])

            if (setting["z_run_name"] == "pyod-loda"):
                model = LODA(contamination=setting["z_nu"], n_bins='auto')

            if (setting["z_run_name"] == "pyod-vae"):
                model = VAE(encoder_neurons=[X_train.shape[0], 64, 32, 16],
                            decoder_neurons=[16, 32, 64, X_train.shape[0]],
                            contamination=setting["z_nu"], random_state=setting["z_random_state"],
                            batch_size=setting["z_batch_size"], epochs=setting["z_epochs"], verbose=0)

            if (setting["z_run_name"] == "pyod-deepsvdd"):
                model = DeepSVDD(contamination=setting["z_nu"], random_state=setting["z_random_state"], 
                                 epochs=setting["z_epochs"], verbose=0)


            model.fit(X_train)

            y_test_pred = model.predict(X_test)
            y_scores = model.decision_function(X_test)

            metrics = calculate_metrics(y_test, y_test_pred)

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)

            if best:
                np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'), y_test_pred, delimiter=',')
                mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'))
                np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'), y_scores, delimiter=',')
                mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'))

            print(f"experiment_pyod {i} metrics {metrics}")
