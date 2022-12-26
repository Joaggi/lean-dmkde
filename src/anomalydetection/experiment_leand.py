import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np
import ast
import tensorflow as tf
from calculate_metrics import calculate_metrics
from calculate_eigs import calculate_eigs
from encoder_decoder_creator import encoder_decoder_creator

import tensorflow as tf

import leand
import anomaly_detector
import adaptive_rff

np.random.seed(42)
tf.random.set_seed(42)


def experiment_leand(X_train, y_train, X_test, y_test, setting, mlflow, best=False):

    #for i, setting in enumerate(settings):
        
        with mlflow.start_run(run_name=setting["z_run_name"]):

            #mlflow.tensorflow.autolog(log_models=False)
            sigma = setting["z_sigma"]
            setting["z_gamma"] = 1/ (2*sigma**2)
            
            setting["z_sequential"] = ast.literal_eval(setting["z_sequential"])
            if not best:
                setting["z_adaptive_input_dimension"] = setting["z_sequential"][-1]
                setting["z_sequential"] = setting["z_sequential"][:-1]
            
            setting["z_layer"] = tf.keras.layers.LeakyReLU()
            setting["z_regularizer"] = tf.keras.regularizers.l1(10e-5)
            
            polinomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(setting["z_base_lr"], \
                setting["z_decay_steps"], setting["z_end_lr"], power=setting["z_power"])
            optimizer = tf.keras.optimizers.Adam(learning_rate=polinomial_decay)  # optimizer

            X = []
            for i in range(len(y_train)):
                if y_train[i] == 0: X.append(X_train[i])
            print(len(X))

            X = np.array(X)

            num_eig = int(setting["z_rff_components"] * setting["z_max_num_eigs"]) 
            
            encoder, decoder = encoder_decoder_creator(input_size=X.shape[1], input_enc=setting["z_adaptive_input_dimension"], \
                        sequential=setting["z_sequential"], layer=setting["z_layer"], regularizer=setting["z_regularizer"])

            aff_dimension = setting["z_adaptive_input_dimension"] + (2 if setting["z_enable_reconstruction_metrics"] else 0)

            leand_alg = leand.Leand(X.shape[1], setting["z_adaptive_input_dimension"], \
                        setting["z_rff_components"], num_eig=num_eig, gamma=setting["z_gamma"], alpha=setting["z_alpha"], \
                        layer=setting["z_layer"], encoder=encoder, decoder=decoder, \
                        enable_reconstruction_metrics=setting["z_enable_reconstruction_metrics"])

            leand_alg.compile(optimizer)
            #eig_vals = leand_alg.set_rho(rho)

            if setting["z_adaptive_fourier_features_enable"] == True:
                autoencoder = anomaly_detector.AnomalyDetector(X.shape[1], setting["z_adaptive_input_dimension"], \
                         layer = setting["z_layer"], \
                         regularizer = setting["z_regularizer"], encoder=encoder, decoder=decoder)

                autoencoder.compile(optimizer=optimizer, loss='mse')

                history = autoencoder.fit(X, X, 
                          epochs=setting["z_autoencoder_epochs"], 
                          batch_size=setting["z_autoencoder_batch_size"],
                          shuffle=True, verbose=0)
                
                encoded_data = autoencoder.encoder(X)

                if setting["z_enable_reconstruction_metrics"]: 
                    reconstruction = autoencoder.decoder(encoded_data)
                    reconstruction = tf.cast(reconstruction, tf.float64)
                    print("Concatenating")
                    reconstruction_loss = tf.keras.losses.binary_crossentropy(X, reconstruction)
                    cosine_similarity = tf.keras.losses.cosine_similarity(X, reconstruction)
                    encoded_kde = tf.keras.layers.Concatenate(axis=1)([encoded_data, tf.reshape(reconstruction_loss, [-1, 1]), tf.reshape(cosine_similarity, [-1,1])]).numpy()
                else:
                    encoded_kde = encoded_data.numpy()
                    print("Not concatenated")

                print(encoded_kde.shape)
                rff_layer = adaptive_rff.fit_transform(setting, encoded_kde)
                leand_alg.encoder = autoencoder.encoder
                leand_alg.decoder = autoencoder.decoder

                leand_alg.fm_x = rff_layer


            history = leand_alg.fit(X, X, 
                      epochs=setting["z_epochs"], 
                      batch_size=setting["z_batch_size"],
                      #validation_data=(test_data, test_data),
                      shuffle=True, verbose=0)



            y_test_pred, _ = leand_alg.predict((X_test, X_test), verbose=0)
            
            g = np.sum(y_test) / len(y_test)
            setting["z_threshold"] = np.percentile(y_test_pred, int(g*100))

            preds = (y_test_pred < setting["z_threshold"]).astype(int)
            metrics = calculate_metrics(y_test, preds, y_test_pred, setting["z_run_name"])

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)

            if best:
                mlflow.log_params({"w_best": best})                
                f = open('./artifacts/'+setting["z_experiment"]+'.npz', 'w')
                np.savez('./artifacts/'+setting["z_experiment"]+'.npz', preds=y_test_pred, scores=y_scores)
                mlflow.log_artifact(('artifacts/'+setting["z_experiment"]+'.npz'))
                f.close()

            print(f"experiment_leand {i} metrics {metrics}")
            print(f"experiment_leand {i} threshold {setting['z_threshold']}")
