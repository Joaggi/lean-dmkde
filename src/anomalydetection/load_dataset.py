import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
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


def load_mat_file(path, algorithm, random_state=42):

    data = scipy.io.loadmat(path)
    features = data["X"]
    labels = data["y"]

    pos_label = 0

    y = convert_dataset(labels, pos_label, algorithm)

    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=random_state, stratify=y)

    return X_train, y_train, X_test, y_test


def load_np(path, algorithm, random_state=42):

    data = np.load(path)
    features = data[:,:-1]
    labels = data[:,-1]

    pos_label = 1

    y = convert_dataset(labels, pos_label, algorithm)

    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=random_state, stratify=y)

    return X_train, y_train, X_test, y_test


def load_dataset(dataset, algorithm, random_state=42):

    if dataset=="spambase":
        path = 'data/'+dataset+".npy"
        return load_np(path, algorithm, random_state)
    else:
        path = 'data/'+dataset+".mat"
        return load_mat_file(path, algorithm, random_state)
