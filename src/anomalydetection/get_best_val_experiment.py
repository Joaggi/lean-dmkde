from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import mlflow_wrapper
from transform_params_to_settings import transform_params_to_settings

def get_best_val_experiment(mlflow, experiment_ids, query, metric_to_select):
 
    runs = mlflow_wrapper.search_runs(mlflow, experiment_ids=experiment_ids, filter_string=query, \
                    run_view_type=ViewType.ACTIVE_ONLY, output_format="pandas")

    best_result = runs.sort_values(metric_to_select, ascending=False).iloc[0]
    print(best_result)

    return transform_params_to_settings(best_result)


