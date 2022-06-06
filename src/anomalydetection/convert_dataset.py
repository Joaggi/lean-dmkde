from check_dataset_for_algorithm import check_dataset_for_algorithm

def convert_dataset(labels, pos_label, algorithm):
    y = []
    type = check_dataset_for_algorithm(algorithm)
    if (type):
        for el in labels:
            y.append(1 if el==pos_label else -1)
    else:
        for el in labels:
            y.append(0 if el==pos_label else 1)
    
    return y


