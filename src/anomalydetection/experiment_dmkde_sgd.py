import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np
import tensorflow as tf
from calculate_metrics import calculate_metrics
from find_best_threshold import find_best_threshold
from calculate_eigs import calculate_eigs

import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)


def experiment_dmkde_sgd(X_train, y_train, X_test, y_test, settings, mlflow, best=False):

    for i, setting in enumerate(settings):
        
        with mlflow.start_run(run_name=setting["z_run_name"]):

            optimizer = tf.keras.optimizers.Adam(learning_rate=setting["z_learning_rate"])

            X = []
            for i in range(len(y_train)):
                if y_train[i] == 0: X.append(X_train[i])
            print(len(X))

            rho, num_eig = calculate_eigs(np.array(X), setting, i)

            qmkde = models.QMDensitySGD(X_train.shape[1], setting["z_rff_components"], num_eig=num_eig, 
                                        gamma=setting["z_gamma"], random_state=setting["z_random_state"])
            qmkde.compile(optimizer)
            eig_vals = qmkde.set_rho(rho)
            qmkde.fit(np.array(X), epochs=setting["z_epochs"], batch_size=setting["z_batch_size"], verbose=0)
            

            y_test_pred = qmkde.predict(X_test)

            if np.isclose(setting["z_threshold"], 0.0, rtol=0.0):
                thresh = find_best_threshold(y_test, y_test_pred)
                setting["z_threshold"] = thresh

            preds = (y_test_pred < setting["z_threshold"]).astype(int)
            metrics = calculate_metrics(y_test, preds)

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)

            if best:
                np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'), preds, delimiter=',')
                mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'))
                np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'), y_test_pred, delimiter=',')
                mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'))

            print(f"experiment_dmkde_sgd {i} metrics {metrics}")
            print(f"experiment_dmkde_sgd {i} threshold {setting['z_threshold']}")