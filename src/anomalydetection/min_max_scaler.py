from sklearn.preprocessing import MinMaxScaler

def min_max_scaler(X_train, *X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    X_transformed_list = []
    for X_test_to_transform in X_test:
        X_transformed_list.append(scaler.transform(X_test_to_transform))
    return [X_train, *X_transformed_list]