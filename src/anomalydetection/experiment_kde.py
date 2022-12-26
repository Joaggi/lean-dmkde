import numpy as np
from sklearn.neighbors import KernelDensity

from calculate_metrics import calculate_metrics


def experiment_kde(X_train, y_train, X_test, y_test, setting, mlflow, best=False):

    #for i, setting in enumerate(settings):

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
            #print("Outlier percentage", g)

            if np.allclose(setting["z_threshold"], 0.0): setting["z_threshold"] = np.percentile(test_score, int(g*100))
            pred = (test_score < setting["z_threshold"]).astype(int)
            
            metrics = calculate_metrics(y_test, pred, test_score, setting["z_run_name"])

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)

            if best:
                mlflow.log_params({"w_best": best})
                f = open('./artifacts/'+setting["z_experiment"]+'.npz', 'w')
                np.savez('./artifacts/'+setting["z_experiment"]+'.npz', preds=y_test_pred, scores=y_scores)
                mlflow.log_artifact(('artifacts/'+setting["z_experiment"]+'.npz'))
                f.close()

            print(f"experiment_kde {i} metrics {metrics}")
            print(f"experiment_kde {i} threshold {setting['z_threshold']}")

          except:
            pass
