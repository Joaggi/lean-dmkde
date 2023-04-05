import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np
import ast
import tensorflow as tf
from calculate_metrics import calculate_metrics
#from calculate_eigs import calculate_eigs
from encoder_decoder_creator import encoder_decoder_creator

import tensorflow as tf

import qadvaeff
import anomaly_detector
import adaptive_rff

np.random.seed(42)
tf.random.set_seed(42)

from mlflow.entities import Param
from mlflow.entities import Metric

from mlflow.tracking import MlflowClient

def experiment_qadvaeff(X_train, y_train, X_test, y_test, setting, mlflow, best=False):

    #for i, setting in enumerate(settings):
        
        with mlflow.start_run(run_name=setting["z_run_name"]) as active_run:
            mlflow_client = MlflowClient()
            print(f"Current Settings: {setting}")

            #mlflow.tensorflow.autolog(log_models=False)
            sigma = setting["z_sigma"]
            setting["z_gamma"] = 1/ (2*sigma**2)

            print("Storing settings")

            mlflow.log_params(params=setting)
            #params=[Param(key=key, value=value) for key, value in setting.items()]
            #mlflow_client.log_batch(run_id=active_run.info.run_id, params=params)
            
            setting["z_sequential"] = ast.literal_eval(setting["z_sequential"])
            setting["z_adaptive_input_dimension"] = setting["z_sequential"][-1]
            setting["z_sequential"] = setting["z_sequential"][:-1]
            
            if setting["z_layer_name"] == "tanh":
                setting["z_layer"] = tf.keras.activations.tanh
            else:
                setting["z_layer"] = tf.keras.layers.LeakyReLU()

            if setting["z_activity_regularizer"] == "l1": 
                setting["z_regularizer"] = tf.keras.regularizers.l1(setting["z_activity_regularizer_value"])
            elif setting["z_activity_regularizer"] == "l2":
                setting["z_regularizer"] = tf.keras.regularizers.l2(setting["z_activity_regularizer_value"])
            else: setting["z_regularizer"] = None

            print('Regularizer:', setting["z_regularizer"])

            setting["z_enable_reconstruction_metrics"] = ast.literal_eval(setting["z_enable_reconstruction_metrics"])
            setting["z_adaptive_fourier_features_enable"] = ast.literal_eval(setting["z_adaptive_fourier_features_enable"])

            #setting["z_rff_components"] = tf.cast(setting["z_rff_components"], tf.float32)
            #setting["z_gamma"] = tf.cast(setting["z_gamma"], tf.float32)
            #setting["z_sigma"] = tf.cast(setting["z_sigma"], tf.float32)
            
            validation_split = 0.2 if setting["z_best"] == False else 0.001



            polinomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(setting["z_base_lr"], \
                setting["z_decay_steps"], setting["z_end_lr"], power=setting["z_power"])
            optimizer = tf.keras.optimizers.Adam(learning_rate=polinomial_decay)  # optimizer

            X = []
            for i in range(len(y_train)):
                if y_train[i] == 0: X.append(X_train[i])
            print(len(X))

            X = np.array(X)

            print("Creating autoencoder")
            num_eig = int(setting["z_rff_components"] * setting["z_max_num_eigs"]) 
            
            encoder, decoder = encoder_decoder_creator(
                autoencoder_type=setting["z_autoencoder_type"],
                input_size=X.shape[1], \
                input_enc=setting["z_adaptive_input_dimension"], \
                sequential=setting["z_sequential"], 
                layer=setting["z_layer"], \
                activity_regularizer_name=setting["z_activity_regularizer"],
                activity_regularizer_value=setting["z_activity_regularizer_value"],
                kernel_regularizer=setting["z_kernel_regularizer"],
                kernel_constraint=setting["z_kernel_contraint"])

            aff_dimension = setting["z_adaptive_input_dimension"] + \
                    (2 if setting["z_enable_reconstruction_metrics"] else 0)

            qadvaeff_alg = qadvaeff.Qadvaeff(X.shape[1], setting["z_adaptive_input_dimension"], \
                        setting["z_rff_components"], num_eig=num_eig, gamma=setting["z_gamma"], \
                        alpha=setting["z_alpha"], \
                        layer=setting["z_layer"], encoder=encoder, decoder=decoder, \
                        enable_reconstruction_metrics=setting["z_enable_reconstruction_metrics"])

            qadvaeff_alg.compile(optimizer)
            #eig_vals = qadvaeff_alg.set_rho(rho)

            print("Training autoencoder")
            if setting["z_autoencoder_is_alone_optimized"] == True or \
                        setting["z_adaptive_fourier_features_enable"] == True:

                autoencoder = anomaly_detector.AnomalyDetector(X.shape[1], \
                           setting["z_adaptive_input_dimension"], \
                         layer = setting["z_layer"], \
                         regularizer = setting["z_regularizer"], encoder=encoder, decoder=decoder)

                autoencoder.compile(optimizer=optimizer, loss='mse')

                history = autoencoder.fit(X, X,
                          validation_split=0.2,
                          epochs=setting["z_autoencoder_epochs"],
                          batch_size=setting["z_autoencoder_batch_size"],
                          shuffle=True, verbose=0)

                encoded_data = autoencoder.encoder(X)
                
                ae_loss_history = [Metric(key="ae_loss", value=value, step=epoch, timestamp=0) 
                           for epoch, value in enumerate(history.history["loss"])] 
                mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=ae_loss_history)
                val_loss_history = [Metric(key="val_loss", value=value, step=epoch, timestamp=0) 
                           for epoch, value in enumerate(history.history["val_loss"])] 
                mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=val_loss_history)
              

            print("Training AFF")
            if setting["z_adaptive_fourier_features_enable"] == True:

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


                rff_layer, adapt_history = adaptive_rff.fit_transform(setting, encoded_kde)
                qadvaeff_alg.encoder = autoencoder.encoder
                qadvaeff_alg.decoder = autoencoder.decoder

                qadvaeff_alg.fm_x = rff_layer
                qadvaeff_alg.fm_x.trainable = False

                adapt_loss_history = [Metric(key="adapt_loss", value=value, step=epoch, timestamp=0) 
                           for epoch, value in enumerate(adapt_history.history["loss"])] 
                mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=adapt_loss_history)
                adapt_val_loss_history = [Metric(key="adapt_val_loss", value=value, step=epoch, timestamp=0)
                           for epoch, value in enumerate(history.history["val_loss"])] 
                mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=adapt_val_loss_history)
              


            if setting["z_autoencoder_is_trainable"] == "False":
                for layer in qadvaeff_alg.encoder.layers:
                    layer.trainable = False
                for layer in qadvaeff_alg.decoder.layers:
                    layer.trainable = False


            print("Training LEAN")
            history = qadvaeff_alg.fit(X, X, 
                      validation_split=validation_split,
                      epochs=setting["z_epochs"], 
                      batch_size=setting["z_batch_size"],
                      shuffle=True, verbose=setting["z_verbose"])

            print('Encoder weights\n', qadvaeff_alg.encoder.layers[-1].get_weights()[0])
            print('Decoder weights\n', qadvaeff_alg.decoder.layers[0].get_weights()[0])
            
            print('storing metrics')
            prob_loss_history = [Metric(key="probs_loss", value=value, step=epoch, timestamp=0)
                       for epoch, value in enumerate(history.history["probs_loss"])] 
            mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=prob_loss_history)
            recon_loss_history = [Metric(key="variational_loss", value=value, step=epoch, timestamp=0)
                       for epoch, value in enumerate(history.history["variational_loss"])] 
            mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=recon_loss_history)
            val_recon_loss_history = [Metric(key="val_variational_loss", value=value, step=epoch, timestamp=0) 
                       for epoch, value in enumerate(history.history["val_variational_loss"])] 
            mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=val_recon_loss_history)
            val_prob_loss_history = [Metric(key="val_probs_loss", value=value, step=epoch, timestamp=0)
                       for epoch, value in enumerate(history.history["val_probs_loss"])] 
            mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=val_prob_loss_history)

           
            print('predicting')
            prob_pred, _, _, _, _ = qadvaeff_alg.predict((X_test, X_test), verbose=0)
            #prob_pred = prob_pred * tf.math.log(prob_pred)
            #y_test_pred = setting["z_alpha"] * prob_pred - (1-setting["z_alpha"]) * reconstruction_pred
            #y_test_pred = setting["z_alpha"] * prob_pred  - (1-setting["z_alpha"]) * reconstruction_pred
            y_test_pred = prob_pred 
            #y_test_pred = reconstruction_pred
            print(y_test)
            g = np.sum(y_test) / len(y_test)
            print(g)

            setting["z_threshold"] = np.percentile(y_test_pred, 100-int(g*100))
            #setting["z_threshold"] = np.percentile(y_test_pred, int(g*100))
            if setting["z_best"] == False:
                mlflow.log_param("z_threshold", setting["z_threshold"])

            #preds = (y_test_pred < setting["z_threshold"]).astype(int)
            preds = (y_test_pred > setting["z_threshold"]).astype(int)
            metrics = calculate_metrics(y_test, preds, y_test_pred, setting["z_run_name"])

            print('storing metrics')
            mlflow.log_metrics(metrics)
            #mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=metrics)

            print('storing artifacts')
            if best:
                f = open('./artifacts/'+setting["z_experiment"]+'.npz', 'w')
                np.savez('./artifacts/'+setting["z_experiment"]+'.npz', preds=preds, scores=y_test_pred)
                mlflow.log_artifact(('artifacts/'+setting["z_experiment"]+'.npz'))
                f.close()

            print(f"experiment_qadvaeff {i} metrics {metrics}")
            print(f"experiment_qadvaeff {i} threshold {setting['z_threshold']}")