import tensorflow as tf
#import pylab as pl
from sklearn.kernel_approximation import RBFSampler
import qmc.tf.layers as layers
import qmc.tf.models as models
import numpy as np 
from sklearn.model_selection import train_test_split
from aff.q_feature_map_adapt_rff import QFeatureMapAdaptRFF 

class DMRFF(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 num_rff,
                 gamma=1,
                 random_state=None):
        super().__init__()
        self.rff_layer = QFeatureMapAdaptRFF(input_dim=dim_x, dim=num_rff, gamma=gamma, random_state=random_state)

    def call(self, inputs):
        x1 = inputs[:, 0, :]
        x2 = inputs[:, 1, :]
        phi1 = self.rff_layer(x1)
        phi2 = self.rff_layer(x2)
        dot = tf.einsum('...i,...i->...', phi1, phi2, optimize='optimal') 
        return dot


