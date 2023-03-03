import itertools
import numpy as np
from operator import itemgetter

def generate_product_dict(original_dict, product_dict):
    
    if not original_dict or not product_dict:
        return {}

    keys = product_dict.keys()
    values = product_dict.values()
    
    if "z_random_search" in original_dict:
        settings = []
        np.random.seed(original_dict["z_random_search_random_state"])
        for i in range(original_dict["z_random_search_iter"]):
            new_dict = original_dict.copy()
            for specific_setting in keys:
                new_dict[specific_setting] = np.random.choice(product_dict[specific_setting], 1)[0]
            settings.append(new_dict)

    else:
        array_product = [{name: dato for name,dato in zip(keys, datos)} \
                         for datos in itertools.product(*values)]
        
        settings = [dict(original_dict, **current_dict) for current_dict in array_product]



    return settings


def add_random_state_to_dict(list_original_dict):

    return [dict(original_dict, **{"z_random_state": 42}) for _, original_dict in enumerate(list_original_dict)]


def generate_dict_with_random_state(original_dict):
    
    return [dict(original_dict, **{"z_random_state": 42})]
