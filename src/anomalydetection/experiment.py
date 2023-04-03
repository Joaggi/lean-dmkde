from load_dataset import load_dataset
from min_max_scaler import min_max_scaler
from sklearn.model_selection import train_test_split
from make_experiment import make_experiment


def experiment(setting, mlflow, best=False):

    algorithm = setting["z_algorithm"]
    dataset = setting["z_dataset"]

    dataset_random_state = setting["z_dataset_random_state"] if "z_dataset_random_state" in setting else None

    X_train, y_train, X_test, y_test = load_dataset(dataset, algorithm, )
    print("Dataset loaded!")
    
    if not best:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
        X_train, X_val, X_test = min_max_scaler(X_train, X_val, X_test)
        print("shape X_train : ", X_train.shape)
        make_experiment(algorithm, X_train, y_train, X_val, y_val, setting, mlflow)

    elif best:
        X_train, X_test = min_max_scaler(X_train, X_test)
        print("BEST shape X_train : ", X_train.shape)
        make_experiment(algorithm, X_train, y_train, X_test, y_test, setting, mlflow, best)



