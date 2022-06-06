from tensorflow import keras
import tensorflow as tf

class AnomalyDetector(keras.Model):
  def __init__(self, input_size, input_enc):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(input_enc, activation="relu")])
    
    self.decoder = tf.keras.Sequential([
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(input_size, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

