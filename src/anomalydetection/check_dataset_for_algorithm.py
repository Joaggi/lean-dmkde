def check_dataset_for_algorithm(algorithm):
    if (algorithm == "oneclass" or algorithm == "isolation" or algorithm == "covariance" or algorithm == "localoutlier"):
        return True
    elif (algorithm.startswith("dmkde") or  algorithm == "qaddemadac" or algorithm == "lake" or algorithm.startswith("pyod")):
        return False
