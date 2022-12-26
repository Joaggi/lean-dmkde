from pyod.models.knn import KNN
from pyod.models.sos import SOS
from pyod.models.copod import COPOD
from pyod.models.loda import LODA
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD

import numpy as np
import ast
from calculate_metrics import calculate_metrics


def experiment_pyod(X_train, y_train, X_test, y_test, setting, mlflow, best=False):

    #for i, setting in enumerate(settings):
        
        with mlflow.start_run(run_name=setting["z_run_name"]):

            model = None
            runname = setting["z_algorithm"]

            if (runname == "pyod-knn"):
                model = KNN(contamination=setting["z_nu"], n_neighbors=setting["z_n_neighbors"])
                
            if (runname == "pyod-sos"):
                model = SOS(contamination=setting["z_nu"], perplexity=setting["z_perplexity"],
                            eps=setting["z_tol"])

            if (runname == "pyod-copod"):
                model = COPOD(contamination=setting["z_nu"])

            if (runname == "pyod-loda"):
                model = LODA(contamination=setting["z_nu"], n_bins='auto')

            if (runname == "pyod-vaebayes"):
                encoder, decoder = ast.literal_eval(setting["z_encoder_neurons"]), ast.literal_eval(setting["z_decoder_neurons"])
                print(setting["z_decoder_neurons"])
                model = VAE(encoder_neurons=np.concatenate((X_train.shape[:1], encoder)),
                            decoder_neurons=np.concatenate((decoder, X_train.shape[:1])),
                            contamination=setting["z_nu"], random_state=setting["z_random_state"],
                            batch_size=setting["z_batch_size"], epochs=setting["z_epochs"], verbose=0)

            if (runname == "pyod-deepsvdd"):
                model = DeepSVDD(contamination=setting["z_nu"], random_state=setting["z_random_state"], 
                                 epochs=setting["z_epochs"], verbose=0)


            model.fit(X_train)

            y_test_pred = model.predict(X_test)
            y_scores = model.decision_function(X_test)
            
            metrics = calculate_metrics(y_test, y_test_pred, y_scores, setting["z_algorithm"])

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)

            if best:
                mlflow.log_params({"w_best": best})
                f = open('./artifacts/'+setting["z_experiment"]+'.npz', 'w')
                np.savez('./artifacts/'+setting["z_experiment"]+'.npz', preds=y_test_pred, scores=y_scores)
                mlflow.log_artifact(('artifacts/'+setting["z_experiment"]+'.npz'))
                f.close()

            nu = setting["z_nu"]
            print(f"experiment: contamination {nu} metrics {metrics}")
