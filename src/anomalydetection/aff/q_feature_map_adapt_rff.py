import tensorflow as tf
#import pylab as pl
from sklearn.kernel_approximation import RBFSampler
import qmc.tf.layers as layers
import qmc.tf.models as models
import numpy as np 
from sklearn.model_selection import train_test_split


class QFeatureMapAdaptRFF(layers.QFeatureMapRFF):
    def __init__(
                 self,
                 gamma_trainable=False,
                 weights_trainable=True,
                 **kwargs
                 ):
        self.g_trainable = gamma_trainable
        self.w_trainable = weights_trainable
        super().__init__(**kwargs)

    def build(self, input_shape):
        print(f"build: dim {self.dim}")
        rbf_sampler = RBFSampler(
            gamma=self.gamma,
            n_components=self.dim,
            random_state=self.random_state)
        x = np.zeros(shape=(1, self.input_dim))
        rbf_sampler.fit(x)
        self.gamma_val = tf.Variable(
            initial_value=self.gamma,
            dtype=tf.float32,
            trainable=self.g_trainable,
            name="rff_gamma")
        self.rff_weights = tf.Variable(
            initial_value=rbf_sampler.random_weights_,
            dtype=tf.float32,
            trainable=self.w_trainable,
            name="rff_weights")
        self.offset = tf.Variable(
            initial_value=rbf_sampler.random_offset_,
            dtype=tf.float32,
            trainable=self.w_trainable,
            name="offset")
        self.built = True

    def call(self, inputs):
        print(f"call: inputs {inputs.shape} rff_weights {self.rff_weights.shape}")
        vals = tf.sqrt(2 * self.gamma_val) * tf.matmul(inputs, self.rff_weights) + self.offset
        vals = tf.cos(vals)
        vals = vals * tf.cast(tf.sqrt(2. / self.dim), tf.float32)
        norms = tf.linalg.norm(vals, axis=-1)
        psi = vals / tf.expand_dims(norms, axis=-1)
        return psi


