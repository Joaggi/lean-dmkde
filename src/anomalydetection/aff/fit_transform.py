import tensorflow as tf
#import pylab as pl
from sklearn.kernel_approximation import RBFSampler
import qmc.tf.layers as layers
import qmc.tf.models as models
import numpy as np 
from sklearn.model_selection import train_test_split

from aff.build_model import build_model 
from aff.build_features import build_features 


def fit_transform(setting, X):
    x_train_rff, x_test_rff = build_features(X, setting["z_adaptive_num_of_samples"], 
                                             setting["z_adaptive_random_samples_enable"])

    return build_model(setting, x_train_rff, x_test_rff)
