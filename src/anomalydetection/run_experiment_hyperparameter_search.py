from mlflow_create_experiment import mlflow_create_experiment 
from generate_product_dict import generate_product_dict, add_random_state_to_dict
from experiment import experiment
import mlflow_wrapper
from mlflow.entities import ViewType
from transform_params_to_settings import transform_params_to_settings
import tensorflow as tf


def get_best_experiment(mlflow, setting):
       
    query = f"params.z_dataset='{setting['z_dataset']}' and params.z_algorithm='{setting['z_algorithm']}'" # and params.z_run_name='NEWW_{setting['z_algorithm']}'
    metrics = ["metrics.aucroc", "metrics.aucpr-an", "metrics.f1_anomalyclass"]
    #metrics = ["metrics.aucpr-an", "metrics.aucroc", "metrics.f1_anomalyclass"]
    experiment_id = mlflow.get_experiment_by_name(setting["z_experiment"]).experiment_id
    runs = mlflow_wrapper.search_runs(mlflow, experiment_id, query, ViewType.ACTIVE_ONLY, output_format="pandas")
    #print([False for i in range(len(metrics))])
    runs.sort_values(metrics, ascending=[False for i in range(len(metrics))], inplace=True)
    best_result = runs.iloc[0]
    
    #print(runs["metrics.aucroc"], runs["metrics.f1_anomalyclass"])
    return transform_params_to_settings(best_result)

    
    
def hyperparameter_search(algorithm, database, parent_path, prod_settings, custom_setting = None):

    print("#-------------------------------------------#")
    setting = {
            "z_algorithm": f"{algorithm}",
            "z_run_name": f"{database}_{algorithm}",
            #"z_run_name": f"{custom_setting['z_experiment']}",
            "z_best": "False"
        }     
    if custom_setting is not None:
            setting = dict(setting, **custom_setting)
     
    mlflow = mlflow_create_experiment(setting["z_experiment"], server=setting["z_mlflow_server"])

    print(f"Original Settings {setting}") 
    print("Starting to create!")

    settings = generate_product_dict(setting, prod_settings)
    settings = add_random_state_to_dict(settings)
    print("Settings created!")

    for i, setting in enumerate(settings):
            experiment(setting = setting, mlflow = mlflow)
        
    best_exp = get_best_experiment(mlflow, setting)    
    threshold = None
    if "z_threshold" in best_exp:
        threshold = best_exp["z_threshold"]
    best_exp = dict(best_exp, **setting)
    if threshold is not None:
        best_exp["z_threshold"] = threshold
    best_exp["z_best"] = True
    return mlflow, best_exp


