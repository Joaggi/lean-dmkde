import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np
import tensorflow as tf
from calculate_metrics import calculate_metrics
from find_best_threshold import find_best_threshold
from calculate_eigs import calculate_eigs
from encoder_decoder_creator import encoder_decoder_creator

import tensorflow as tf

import leand
import anomaly_detector
import adaptive_rff

np.random.seed(42)
tf.random.set_seed(42)


def experiment_leand(X_train, y_train, X_test, y_test, settings, mlflow, best=False):

    for i, setting in enumerate(settings):
        
        with mlflow.start_run(run_name=setting["z_run_name"]):

            
            polinomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(setting["z_base_lr"], \
                setting["z_decay_steps"], setting["z_end_lr"], power=setting["z_power"])
            optimizer = tf.keras.optimizers.Adam(learning_rate=polinomial_decay)  # optimizer

            X = []
            for i in range(len(y_train)):
                if y_train[i] == 0: X.append(X_train[i])
            print(len(X))

            X = np.array(X)

            
            encoder, decoder = encoder_decoder_creator(input_size=X.shape[1], input_enc=setting["z_adaptive_input_dimension"], \
                        sequential=setting["z_sequential"], layer=setting["z_layer"], regularizer=setting["z_regularizer"])

            autoencoder = anomaly_detector.AnomalyDetector(X.shape[1], setting["z_adaptive_input_dimension"], \
                     layer = setting["z_layer"], \
                     regularizer = setting["z_regularizer"], encoder=encoder, decoder=decoder)

            autoencoder.compile(optimizer=optimizer, loss='mse')

            history = autoencoder.fit(X, X, 
                      epochs=setting["z_autoencoder_epochs"], 
                      batch_size=setting["z_autoencoder_batch_size"],
                      shuffle=True)
            
            encoded_data = autoencoder.encoder(X)

            reconstruction = autoencoder.decoder(encoded_data)
            reconstruction = tf.cast(reconstruction, tf.float64)

            reconstruction_loss = (1-setting["z_alpha"]) * tf.keras.losses.binary_crossentropy(X, reconstruction)
            
            cosine_similarity = tf.keras.losses.cosine_similarity(X, reconstruction)

            encoded_kde = tf.keras.layers.Concatenate(axis=1)([encoded_data, tf.reshape(reconstruction_loss, [-1, 1]), tf.reshape(cosine_similarity, [-1,1])]).numpy()


            rff_layer = adaptive_rff.fit_transform(setting, encoded_kde)

            dmkde = models.QMDensity(rff_layer, setting["z_dim_rff"])
            dmkde.compile()

            dmkde.predict(X)


            test_score = []
            for i in range(len(X_test)):
                score = kde.score_samples(X_test[i])
                test_score.append(score)
            test_labels = np.concatenate(test_labels,axis=0)
            test_score = np.concatenate(test_score,axis=0)


            g = np.sum(test_labels==1) / len(test_labels)

            thresh = np.percentile(test_score, int(g*100))
            pred = (test_score < thresh).astype(int)
            gt = test_labels.astype(int)

            metrics = calculate_metrics(gt, pred, test_score, setting["z_run_name"])



            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)
            [mlflow.log_metric("loss", metric, i) for metric, i in enumerate(history.history["loss"])]
            [mlflow.log_metric("reconstruction_loss", metric, i) for metric, i in enumerate(history.history["reconstruction_loss"])]
            [mlflow.log_metric("probs_loss", metric, i) for metric, i in enumerate(history.history["probs_loss"])]

            if best:
                np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'), preds, delimiter=',')
                mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'))
                np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'), y_test_pred, delimiter=',')
                mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'))

            print(f"experiment_dmkde_sgd {i} metrics {metrics}")
            print(f"experiment_dmkde_sgd {i} threshold {setting['z_threshold']}")
