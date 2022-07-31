from tensorflow import keras
import tensorflow as tf

class AnomalyDetector(keras.Model):
  def __init__(self, input_size, input_enc, layer = tf.keras.layers.LeakyReLU(), \
        regularizer = tf.keras.regularizers.l1(10e-5), encoder=None, decoder=None):

    super(AnomalyDetector, self).__init__()

    if encoder == None:
        self.encoder = tf.keras.Sequential([
          keras.layers.Dense(64, activation=layer, activity_regularizer=regularizer),
          keras.layers.Dense(32, activation=layer, activity_regularizer=regularizer),
          keras.layers.Dense(input_enc, activation=layer, activity_regularizer=regularizer)])
    else:
        self.encoder = encoder
    
    if decoder == None:
        self.decoder = tf.keras.Sequential([
          keras.layers.Dense(32, activation=layer, activity_regularizer=regularizer),
          keras.layers.Dense(64, activation=layer, activity_regularizer=regularizer),
          keras.layers.Dense(input_size, activation="sigmoid")])
    else:
        self.decoder = decoder 
         
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

