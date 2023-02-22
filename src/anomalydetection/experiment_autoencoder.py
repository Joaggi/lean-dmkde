import qmc.tf.layers as layers
import qmc.tf.models as models

import ast
import numpy as np
import tensorflow as tf
from calculate_metrics import calculate_metrics
#from calculate_eigs import calculate_eigs
from encoder_decoder_creator import encoder_decoder_creator

import tensorflow as tf

import anomaly_detector


np.random.seed(42)
tf.random.set_seed(42)


def experiment_ae(X_train, y_train, X_test, y_test, setting, mlflow, best=False):

    #for i, setting in enumerate(settings):

        with mlflow.start_run(run_name=setting["z_run_name"]):


            polinomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(setting["z_base_lr"], \
                setting["z_decay_steps"], setting["z_end_lr"], power=setting["z_power"])
            optimizer = tf.keras.optimizers.Adam(learning_rate=polinomial_decay)  # optimizer

            X = []
            for i in range(len(y_train)):
                if y_train[i] == 0: X.append(X_train[i])
            print(len(X))

            X = np.array(X)

            setting["z_sequential"] = ast.literal_eval(setting["z_sequential"])
            setting["z_layer"] = tf.keras.layers.Activation(setting["z_layername"])
            setting["z_regularizer"] = tf.keras.regularizers.l1(10e-5)

            encoder, decoder = encoder_decoder_creator(input_size=X.shape[1], input_enc=setting["z_sequential"][-1], \
                        sequential=setting["z_sequential"][:-1], layer=setting["z_layer"], regularizer=setting["z_regularizer"])

            autoencoder = anomaly_detector.AnomalyDetector(X.shape[1], setting["z_sequential"][-1], \
                     layer = setting["z_layer"], \
                     regularizer = setting["z_regularizer"], encoder=encoder, decoder=decoder)

            autoencoder.compile(optimizer=optimizer, loss='mse')

            history = autoencoder.fit(X, X, 
                      epochs=setting["z_autoencoder_epochs"], 
                      batch_size=setting["z_autoencoder_batch_size"],
                      shuffle=True, verbose=0)

            encoded_data = autoencoder.encoder(X)

            reconstruction = autoencoder.decoder(encoded_data)
            reconstruction = tf.cast(reconstruction, tf.float64)
            reconstruction_loss = [ (np.linalg.norm(reconstruction[i]-X[i]))**2 for i in range(X.shape[0]) ]

            nu = np.sum(y_test) / len(y_test)
            if np.allclose(setting["z_threshold"], 0.0): setting["z_threshold"] = np.percentile(reconstruction_loss, 100*(1-nu))

            reconstruction_test = autoencoder.predict((X_test, X_test))
            reconstruction_error = [ (np.linalg.norm(reconstruction_test[i]-X_test[i]))**2 for i in range(X_test.shape[0]) ]

            #print(reconstruction_error)
            preds = (reconstruction_error > setting["z_threshold"]).astype(int)

            metrics = calculate_metrics(y_test, preds, reconstruction_error, setting["z_run_name"])

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)
            
            if best:
                mlflow.log_params({"w_best": best})
                f = open('./artifacts/'+setting["z_experiment"]+'.npz', 'w')
                np.savez('./artifacts/'+setting["z_experiment"]+'.npz', preds=preds, scores=reconstruction_error)
                mlflow.log_artifact(('artifacts/'+setting["z_experiment"]+'.npz'))
                f.close()
            
            print(f"experiment_ae {i} metrics {metrics}")
            print(f"experiment_ae {i} threshold {setting['z_threshold']}")
