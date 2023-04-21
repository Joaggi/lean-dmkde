import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np
from calculate_metrics import calculate_metrics
from sklearn.kernel_approximation import RBFSampler

import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

from sigma_calculator import sigma_calculator
import adaptive_rff


from mlflow.entities import Param
from mlflow.entities import Metric

from mlflow.tracking import MlflowClient

def experiment_addmkde(X_train, y_train, X_test, y_test, setting, mlflow, best=False):

    with mlflow.start_run(run_name=setting["z_run_name"]) as active_run:

        mlflow_client = MlflowClient()


        X = []
        for j in range(len(y_train)):
           if y_train[j] == 0: X.append(X_train[j])
        x_train = np.array(X)

        setting["z_adaptive_input_dimension"] = x_train.shape[1]

        setting["z_sigma"] = sigma_calculator(x_train, percentile=setting["z_percentile"], multiplier=setting["z_multiplier"])
        setting["z_gamma"] = 1/ (2*setting["z_sigma"]**2)


        mlflow.log_params(setting)

        print("Training AFF")
        if "z_adaptive_fourier_features_enable" in setting and setting["z_adaptive_fourier_features_enable"] == "True":

            rff_layer, adapt_history = adaptive_rff.fit_transform(setting, x_train)

            fm_x = rff_layer
            fm_x.trainable = False

            ##adapt_loss_history = [Metric(key="adapt_loss", value=value, step=epoch, timestamp=0) 
            ##           for epoch, value in enumerate(adapt_history.history["loss"])] 
            ##mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=adapt_loss_history)
            ##adapt_val_loss_history = [Metric(key="adapt_val_loss", value=value, step=epoch, timestamp=0)
            ##           for epoch, value in enumerate(adapt_history.history["val_loss"])] 
            ##mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=adapt_val_loss_history)

        else:
            fm_x = adaptive_rff.QFeatureMapAdaptRFF(input_dim=x_train.shape[1], dim=setting["z_rff_components"], gamma=setting["z_gamma"], random_state=setting["z_random_state"])
            fm_x.build(input_shape=x_train.shape[1])
            fm_x.trainable = False
   
        qmd = models.QMDensity(fm_x, setting["z_rff_components"])
        qmd.compile()
        qmd.fit(np.array(X), epochs=1, batch_size=setting["z_batch_size"], verbose=1)
        
        y_scores = qmd.predict(X_test)

        g = np.sum(y_test) / len(y_test)
        print("Outlier percentage", g)

        if np.allclose(setting["z_threshold"], 0.0): setting["z_threshold"] = np.percentile(y_scores, int(g*100))
        preds = (y_scores < setting["z_threshold"]).astype(int)
        metrics = calculate_metrics(y_test, preds, y_scores, setting["z_run_name"])

        mlflow.log_metrics(metrics)

        if best:
            f = open('./artifacts/'+setting["z_experiment"]+'.npz', 'w')
            np.savez('./artifacts/'+setting["z_experiment"]+'.npz', preds=preds, scores=y_scores)
            mlflow.log_artifact(('artifacts/'+setting["z_experiment"]+'.npz'))
            f.close()

        print(f"experiment_dmkde_adp {setting['z_gamma']} metrics {metrics}")
        print(f"experiment_dmkde_adp {setting['z_gamma']} threshold {setting['z_threshold']}")
