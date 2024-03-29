import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np
import ast
import tensorflow as tf
from calculate_metrics import calculate_metrics
#from calculate_eigs import calculate_eigs
from encoder_decoder_creator import encoder_decoder_creator

import tensorflow as tf

import leand
import anomaly_detector
import adaptive_rff

np.random.seed(42)
tf.random.set_seed(42)

from mlflow.entities import Param
from mlflow.entities import Metric

from mlflow.tracking import MlflowClient

from sigma_calculator import sigma_calculator

def experiment_leand(X_train, y_train, X_test, y_test, setting, mlflow, best=False):

    #for i, setting in enumerate(settings):
        
        with mlflow.start_run(run_name=setting["z_run_name"]) as active_run:
            mlflow_client = MlflowClient()
            print(f"Current Settings: {setting}")

            #mlflow.tensorflow.autolog(log_models=False)

            #params=[Param(key=key, value=value) for key, value in setting.items()]
            #mlflow_client.log_batch(run_id=active_run.info.run_id, params=params)


            print("Storing settings")
            mlflow.log_params(params=setting)

            
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

            print("Training autoencoder")

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
            

            if setting["z_log_loss"] == True:
                ae_loss_history = [Metric(key="ae_loss", value=value, step=epoch, timestamp=0) 
                           for epoch, value in enumerate(history.history["loss"])] 
                mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=ae_loss_history)
                val_loss_history = [Metric(key="val_loss", value=value, step=epoch, timestamp=0) 
                           for epoch, value in enumerate(history.history["val_loss"])] 
                mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=val_loss_history)

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

            #if "z_best" not in setting or setting["z_best"] == 'False':
            #    setting["z_sigma"] = sigma_calculator(encoded_kde, percentile=setting["z_percentile"], multiplier=setting["z_multiplier"])
            #    setting["z_gamma"] = 1/ (2*setting["z_sigma"]**2)
            #    mlflow.log_param("z_sigma", setting["z_sigma"])
            #    mlflow.log_param("z_gamma", setting["z_gamma"])

            setting["z_gamma"] = 1/ (2*setting["z_sigma"]**2)


            leand_alg = leand.Leand(X.shape[1], setting["z_adaptive_input_dimension"], \
                        setting["z_rff_components"], num_eig=num_eig, gamma=setting["z_gamma"], \
                        alpha=setting["z_alpha"], \
                        layer=setting["z_layer"], encoder=encoder, decoder=decoder, \
                        enable_reconstruction_metrics=setting["z_enable_reconstruction_metrics"])

            leand_alg.compile(optimizer)
            #eig_vals = leand_alg.set_rho(rho)


            print("Training AFF")
            if setting["z_adaptive_fourier_features_enable"] == True:


                rff_layer, adapt_history = adaptive_rff.fit_transform(setting, encoded_kde)
                leand_alg.encoder = autoencoder.encoder
                leand_alg.decoder = autoencoder.decoder

                leand_alg.fm_x = rff_layer
                leand_alg.fm_x.trainable = False
                #leand_alg.fm_x.gamma_val.trainable = False
                #leand_alg.fm_x.rff_weights.trainable = False
                #leand_alg.fm_x.offset.trainable = False


                if setting["z_log_loss"] == True:
                    adapt_loss_history = [Metric(key="adapt_loss", value=value, step=epoch, timestamp=0) 
                               for epoch, value in enumerate(adapt_history.history["loss"])] 
                    mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=adapt_loss_history)
                    adapt_val_loss_history = [Metric(key="adapt_val_loss", value=value, step=epoch, timestamp=0)
                               for epoch, value in enumerate(adapt_history.history["val_loss"])] 
                    mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=adapt_val_loss_history)
              


            if setting["z_autoencoder_is_trainable"] == "False":
                for layer in leand_alg.encoder.layers:
                    layer.trainable = False
                for layer in leand_alg.decoder.layers:
                    layer.trainable = False


            print("Training LEAN")
            history = leand_alg.fit(X, X, 
                      validation_split=validation_split,
                      epochs=setting["z_epochs"], 
                      batch_size=setting["z_batch_size"],
                      shuffle=True, verbose=setting["z_verbose"])

            print('Encoder weights\n', leand_alg.encoder.layers[-1].get_weights()[0])
            print('Decoder weights\n', leand_alg.decoder.layers[0].get_weights()[0])
            
            print('storing metrics')

            if setting["z_log_loss"] == True:
                prob_loss_history = [Metric(key="probs_loss", value=value, step=epoch, timestamp=0)
                           for epoch, value in enumerate(history.history["probs_loss"])] 
                mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=prob_loss_history)
                recon_loss_history = [Metric(key="recon_loss", value=value, step=epoch, timestamp=0)
                           for epoch, value in enumerate(history.history["reconstruction_loss"])] 
                mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=recon_loss_history)
                val_recon_loss_history = [Metric(key="val_recon_loss", value=value, step=epoch, timestamp=0) 
                           for epoch, value in enumerate(history.history["val_reconstruction_loss"])] 
                mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=val_recon_loss_history)
                val_prob_loss_history = [Metric(key="val_probs_loss", value=value, step=epoch, timestamp=0)
                           for epoch, value in enumerate(history.history["val_probs_loss"])] 
                mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=val_prob_loss_history)

           
            print('predicting')
            prob_pred, reconstruction_pred = leand_alg.predict((X_test, X_test), verbose=0)
            #prob_pred = prob_pred * tf.math.log(prob_pred)
            y_test_pred = setting["z_alpha"] * prob_pred - (1-setting["z_alpha"]) * reconstruction_pred
            #y_test_pred = setting["z_alpha"] * prob_pred  + (1-setting["z_alpha"]) * reconstruction_pred
            #y_test_pred = prob_pred 
            #y_test_pred = reconstruction_pred
            print(y_test)
            g = np.sum(y_test) / len(y_test)
            print(g)

            setting["z_threshold"] = np.percentile(y_test_pred, 100-int(g*100))
            #setting["z_threshold"] = np.percentile(y_test_pred, int(g*100))
            if setting["z_best"] == 'False':
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

            print(f"experiment_leand {i} metrics {metrics}")
            print(f"experiment_leand {i} threshold {setting['z_threshold']}")
