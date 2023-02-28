import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Layer, InputSpec
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers, activations, initializers, constraints, Sequential
from keras import backend as K
from keras.constraints import UnitNorm, Constraint


class WeightsOrthogonalityConstraint(tf.keras.constraints.Constraint):
    
    def __init__(self, encoding_dim, weightage=1.0, axis=0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis
        
    def weights_orthogonality(self, w):
        if self.axis == 1:
            w = tf.transpose(w)
        if self.encoding_dim > 1:
            m = tf.matmul(tf.transpose(w), w) - tf.eye(self.encoding_dim)
            return self.weightage * tf.sqrt(tf.reduce_sum(tf.square(m)))
        else:
            m = tf.reduce_sum(tf.square(w)) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)

