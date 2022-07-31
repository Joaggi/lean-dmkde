import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np
import tensorflow as tf
from calculate_metrics import calculate_metrics
from find_best_threshold import find_best_threshold
from calculate_eigs import calculate_eigs

import tensorflow as tf

import qadvaeff
import anomaly_detector
import adaptive_rff

np.random.seed(42)
tf.random.set_seed(42)


def experiment_qadvaeff(X_train, y_train, X_test, y_test, settings, mlflow, best=False):

    for i, setting in enumerate(settings):
        
        with mlflow.start_run(run_name=setting["z_run_name"]):

            sigma = setting["z_sigma"]
            setting["z_gamma"] = 1/ (2*sigma**2)

            
            polinomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(setting["z_base_lr"], \
                setting["z_decay_steps"], setting["z_end_lr"], power=setting["z_power"])
            optimizer = tf.keras.optimizers.Adam(learning_rate=polinomial_decay)  # optimizer

            X = []
            for i in range(len(y_train)):
                if y_train[i] == 0: X.append(X_train[i])
            print(len(X))

            X = np.array(X)

            rho, num_eig = calculate_eigs(X, setting, i)


            qadvaeff_alg = qadvaeff.Qadvaeff(X.shape[1], setting["z_adaptive_input_dimension"], setting["z_rff_components"], num_eig=num_eig, gamma=setting["z_gamma"])

            qadvaeff_alg.compile(optimizer)
            #eig_vals = qadvaeff_alg.set_rho(rho)

            if setting["z_adaptive_fourier_features_enable"] == True:
                autoencoder = anomaly_detector.AnomalyDetector(X.shape[1], setting["z_adaptive_input_dimension"])
                autoencoder.compile(optimizer=optimizer, loss='mae')
                history = autoencoder.fit(X, X, 
                          epochs=setting["z_autoencoder_epochs"], 
                          batch_size=setting["z_autoencoder_batch_size"],
                          shuffle=True)
                
                encoded_data = autoencoder.encoder(X).numpy()
                rff_layer = adaptive_rff.fit_transform(setting, encoded_data)
                qadvaeff_alg.fm_x = rff_layer


            _ = qadvaeff_alg.fit(X, X, 
                      epochs=setting["z_epochs"], 
                      batch_size=setting["z_batch_size"],
                      #validation_data=(test_data, test_data),
                      shuffle=True)

            

            y_test_pred, _, _, _, _ = qadvaeff_alg.predict((X_test, X_test))

            if np.isclose(setting["z_threshold"], 0.0, rtol=0.0):
                thresh = find_best_threshold(y_test, y_test_pred)
                setting["z_threshold"] = thresh

            preds = (y_test_pred < setting["z_threshold"]).astype(int)
            metrics = calculate_metrics(y_test, preds, y_test_pred, setting["z_run_name"])

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)

            if best:
                np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'), preds, delimiter=',')
                mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'))
                np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'), y_test_pred, delimiter=',')
                mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'))

            print(f"experiment_dmkde_sgd {i} metrics {metrics}")
            print(f"experiment_dmkde_sgd {i} threshold {setting['z_threshold']}")
