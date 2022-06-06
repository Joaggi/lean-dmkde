import itertools
import numpy as np
from operator import itemgetter

def generate_product_dict(original_dict, product_dict):
    
    if not original_dict or not product_dict:
        return {}

    keys = product_dict.keys()
    values = product_dict.values()
 
    array_product = [{name: dato for name,dato in zip(keys, datos)} for datos in itertools.product(*values)]
    
    settings = [dict(original_dict, **current_dict) for current_dict in array_product]

    
    if "z_random_search" in original_dict and  original_dict["z_random_search"] == True:
        print("Random search enable")
        settings_length = len(settings)
        np.random.seed(original_dict["z_random_search_random_state"])
        random_list = np.random.choice(settings_length, original_dict["z_random_search_iter"] if settings_length > 20 else settings_length).tolist()

        settings =  list(itemgetter(*random_list)(settings))

    return settings


def add_random_state_to_dict(list_original_dict):

    return [dict(original_dict, **{"z_random_state": 42}) for _, original_dict in enumerate(list_original_dict)]


def generate_dict_with_random_state(original_dict):
    
    return [dict(original_dict, **{"z_random_state": 42})]