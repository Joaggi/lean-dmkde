import tensorflow as tf
#import pylab as pl
from sklearn.kernel_approximation import RBFSampler
import qmc.tf.layers as layers
import qmc.tf.models as models
import numpy as np 
from sklearn.model_selection import train_test_split


def build_features(X, num_samples, random_samples_enable):
    X_train, X_test = train_test_split(X)

    if random_samples_enable == "True":
        random_samples = np.random.uniform(-3*X_train.std() + X_train.min(), 3*X_train.std() +
                                     X_train.max(),size=(X_train.shape[0]*2, X_train.shape[1]))

        X_train_random = np.concatenate([X_train, random_samples], axis=0)
    else:
        X_train_random = X_train

    rnd_idx1 = np.random.randint(X_train_random.shape[0],size=(num_samples, ))
    rnd_idx2 = np.random.randint(X_train_random.shape[0],size=(num_samples, ))
    x_train_rff = np.concatenate([X_train_random[rnd_idx1][:, np.newaxis, ...], 
                              X_train_random[rnd_idx2][:, np.newaxis, ...]], 
                             axis=1)
    #dists = np.linalg.norm(x_train_rff[:, 0, ...] - x_train_rff[:, 1, ...], axis=1)
    #print(dists.shape)
    #pl.hist(dists)
    #print(np.quantile(dists, 0.001))
    rnd_idx1 = np.random.randint(X_test.shape[0],size=(num_samples, ))
    rnd_idx2 = np.random.randint(X_test.shape[0],size=(num_samples, ))
    x_test_rff = np.concatenate([X_test[rnd_idx1][:, np.newaxis, ...], 
                              X_test[rnd_idx2][:, np.newaxis, ...]], 
                             axis=1)
    print("AFF sizes:", x_train_rff.shape, '-', x_test_rff.shape)
    return x_train_rff, x_test_rff
