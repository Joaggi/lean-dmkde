def check_dataset_for_algorithm(algorithm):
    if (algorithm == "oneclass" or algorithm == "isolation" or algorithm == "covariance" or algorithm == "localoutlier"):
        return True
    else:
        return False
