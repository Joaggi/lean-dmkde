import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np
import tensorflow as tf
from calculate_metrics import calculate_metrics
from find_best_threshold import find_best_threshold
from calculate_eigs import calculate_eigs

from encoder_decoder_creator import encoder_decoder_creator
import tensorflow as tf

import qadvaeff
import variational_autoencoder
import adaptive_rff

np.random.seed(42)
tf.random.set_seed(42)


def experiment_qadvaeff(X_train, y_train, X_test, y_test, settings, mlflow, best=False):

    for i, setting in enumerate(settings):
        
        with mlflow.start_run(run_name=setting["z_run_name"]):
            #mlflow.tensorflow.autolog(log_models=False)
            sigma = setting["z_sigma"]
            setting["z_gamma"] = 1/ (2*sigma**2)

            setting["z_adaptive_input_dimension"] = setting["z_sequential"][-1]
            setting["z_sequential"] = setting["z_sequential"][:-1]
            
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

            qadvaeff_alg = qadvaeff.Qadvaeff(X.shape[1], setting["z_adaptive_input_dimension"], \
                        setting["z_rff_components"], num_eig=num_eig, gamma=setting["z_gamma"], alpha=setting["z_alpha"], \
                        layer=setting["z_layer"], encoder=encoder, decoder=decoder, \
                        enable_reconstruction_metrics=setting["z_enable_reconstruction_metrics"])

            qadvaeff_alg.compile(optimizer)


            if setting["z_adaptive_fourier_features_enable"] == True:
                autoencoder = variational_autoencoder.VariationalAutoencoder(X.shape[1], setting["z_adaptive_input_dimension"],
                        encoder=encoder, decoder=decoder)
                autoencoder.compile(optimizer=optimizer, loss='mae')
                history = autoencoder.fit(X, X, 
                          epochs=setting["z_autoencoder_epochs"], 
                          batch_size=setting["z_autoencoder_batch_size"],
                          shuffle=True)
                encoded_data = autoencoder.encoder(X).numpy()

                if setting["z_enable_reconstruction_metrics"]: 
                    mean, log_var = autoencoder.encode(encoded_data)

                    print(f"call:reparameterize mean {mean} log_var {log_var}")
                    z = autoencoder.sampling([mean, log_var])
         
                    print("call: decode")
                    reconstruction = autoencoder.decode(z)

                    reconstruction = tf.cast(reconstruction, tf.float64)

                    
                    reconstruction_loss = tf.keras.losses.binary_crossentropy(X, reconstruction)
                    
                    cosine_similarity = tf.keras.losses.cosine_similarity(X, reconstruction)

                    encoded_kde = tf.keras.layers.Concatenate(axis=1)([encoded_data, tf.reshape(reconstruction_loss, [-1, 1]), tf.reshape(cosine_similarity, [-1,1])]).numpy()
                else:
                    encoded_kde = encoded_data
                
                datos = np.concatenate([np.random.uniform(-1.5, 1.5, size=(1000, encoded_kde.shape[1])), encoded_kde])

                rff_layer = adaptive_rff.fit_transform(setting, datos)
                qadvaeff_alg.fm_x = rff_layer
                qadvaeff_alg.encoder = autoencoder.encoder
                qadvaeff_alg.dense = autoencoder.dense
                qadvaeff_alg.decoder = autoencoder.decoder

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
