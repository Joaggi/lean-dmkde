import numpy as np
from sklearn.covariance import EllipticEnvelope

from calculate_metrics import calculate_metrics


def experiment_covariance(X_train, y_train, X_test, y_test, setting, mlflow, best=False):
    
#    for i, setting in enumerate(settings):

        with mlflow.start_run(run_name=setting["z_run_name"]):

            model = EllipticEnvelope(contamination=setting["z_nu"], random_state=setting["z_random_state"])
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
