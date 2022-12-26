from mlflow_create_experiment import mlflow_create_experiment 
from generate_product_dict import generate_product_dict, add_random_state_to_dict
from experiment import experiment
import mlflow_wrapper
from mlflow.entities import ViewType
from transform_params_to_settings import transform_params_to_settings



def get_best_experiment(mlflow, setting):
       
    query = f"params.z_dataset='{setting['z_dataset']}'" # and params.z_run_name='NEWW_{setting['z_algorithm']}'
    metric = "metrics.f1_score"
    experiment_id = mlflow.get_experiment_by_name(setting["z_run_name"]).experiment_id
    runs = mlflow_wrapper.search_runs(mlflow, experiment_id, query, ViewType.ACTIVE_ONLY, output_format="pandas")
    best_result = runs.sort_values(metric, ascending=False).iloc[0]
    
    print(best_result)
    return transform_params_to_settings(best_result)

    
    
def hyperparameter_search(algorithm, database, parent_path, prod_settings, custom_setting = None):
       
    print("#-------------------------------------------#")
    setting = {
        "z_algorithm": f"{algorithm}",
        "z_experiment": f"{database}_{algorithm}",
        "z_run_name": f"NEWW_{algorithm}"
    }
    
    if custom_setting is not None:
        setting = dict(setting, **custom_setting)
          
    mlflow = mlflow_create_experiment(setting["z_run_name"])

    settings = generate_product_dict(setting, prod_settings)
    settings = add_random_state_to_dict(settings)
    print("Settings created!")

    #print(len(settings))
    for i, setting in enumerate(settings):
        experiment(setting = setting, mlflow = mlflow)
    
    best_exp = get_best_experiment(mlflow, setting)    
    return mlflow, best_exp

