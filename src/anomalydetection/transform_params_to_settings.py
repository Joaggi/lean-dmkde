def transform_params_to_settings(best_result):
    keys = best_result.keys()
    filter = keys.str.match(r'(^params\.*)')
    best_params = best_result[keys[filter]]
    keys = best_params.keys()
    new_keys = keys.str.replace('params.','')
    best_params.index = new_keys

    return best_params


