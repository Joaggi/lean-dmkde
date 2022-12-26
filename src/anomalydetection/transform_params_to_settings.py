import pandas as pd

def transform_params_to_settings(best_result):
    keys = best_result.keys()
    filter = keys.str.match(r'(^params\.*)')
    best_params = best_result[keys[filter]]
    
    keys = best_params.keys()
    vals = best_params.values
    new_keys = keys.str.replace('params.','')
    best_params.index = new_keys

    best_params = best_params.apply(pd.to_numeric, errors='ignore')

    return best_params


