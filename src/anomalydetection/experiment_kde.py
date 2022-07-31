from typing import get_type_hints
import numpy as np 

from sklearn.neighbors import KernelDensity

from calculate_metrics import calculate_metrics


np.random.seed(42)


def experiment_lake(X_train, y_train, X_test, y_test, settings, mlflow, best=False):

    for i, setting in enumerate(settings):
        #print(f"experiment_dmkdc {i} threshold {setting['z_threshold']}")
        with mlflow.start_run(run_name=setting["z_run_name"]):

            kde = KernelDensity(kernel='gaussian', bandwidth=0.000001).fit(X_train)
            score = kde.score_samples(X_train)

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

            if best:
                np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'), pred, delimiter=',')
                mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'))
                np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'), test_score, delimiter=',')
                mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'))

            print(f"experiment_lake {i} ratio {setting['z_ratio']}")
            print(f"experiment_lake {i} metrics {metrics}")
