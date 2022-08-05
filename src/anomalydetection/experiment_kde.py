import numpy as np
from sklearn.neighbors import KernelDensity

from calculate_metrics import calculate_metrics
from find_best_threshold import find_best_threshold


def experiment_kde(X_train, y_train, X_test, y_test, settings, mlflow, best=False):

    for i, setting in enumerate(settings):

        with mlflow.start_run(run_name=setting["z_run_name"]):

          try:
            X = []
            for i in range(len(y_train)):
                if y_train[i] == 0: X.append(X_train[i])

            model = KernelDensity(kernel=setting["z_kernel_fun"], bandwidth=setting["z_bandwidth"])
            model.fit(np.array(X))

            test_score = []
            for i in range(len(X_test)):
                score = model.score_samples(np.array(X_test[i]).reshape(1,-1))
                test_score.append(score)
            
            test_score = np.concatenate(test_score,axis=0)

            g = np.sum(y_test) / len(y_test)
            print("Outlier percentage", g)

            thresh = np.percentile(test_score, int(g*100))
            pred = (test_score < thresh).astype(int)
            setting["z_threshold"] = thresh
            
            metrics = calculate_metrics(y_test, pred, test_score, setting["z_run_name"])

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)

            if best:
                np.savetxt(('/home/oabustosb/mlflow-persistence/artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'), pred, delimiter=',')
                mlflow.log_artifact(('/home/oabustosb/mlflow-persistence/artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'))
                np.savetxt(('/home/oabustosb/mlflow-persistence/artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'), test_score, delimiter=',')
                mlflow.log_artifact(('/home/oabustosb/mlflow-persistence/artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'))

            print(f"experiment_kde {i} metrics {metrics}")
            print(f"experiment_kde {i} threshold {setting['z_threshold']}")

          except:
            pass
