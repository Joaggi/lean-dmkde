import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Layer, InputSpec
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers, activations, initializers, constraints, Sequential
from keras import backend as K
from keras.constraints import UnitNorm, Constraint


class UncorrelatedFeaturesConstraint(tf.keras.constraints.Constraint):
    def __init__(self, encoding_dim, weightage=1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage

    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - tf.reduce_mean(x[:, i]))
        
        x_centered = tf.stack(x_centered_list)
        covariance = tf.matmul(x_centered, x_centered, transpose_b=True) / tf.cast(x_centered.get_shape()[0], tf.float32)
        
        return covariance

    def uncorrelated_feature(self, x):
        if self.encoding_dim <= 1:
            return 0.0
        else:
            covariance = self.get_covariance(x)
            output = tf.reduce_sum(tf.square(covariance - tf.math.multiply(covariance, tf.eye(self.encoding_dim))))
            return output

    def __call__(self, x):
        return self.weightage * self.uncorrelated_feature(x)
