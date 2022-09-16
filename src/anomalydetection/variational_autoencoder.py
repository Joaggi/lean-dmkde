from tensorflow import keras
import qmc.tf.layers as layers
import qmc.tf.models as models
import tensorflow as tf
import numpy as np

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

class Sampling(keras.layers.Layer):
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VariationalAutoencoder(keras.Model):
  """
    A Variatonal Autoencoder Anomaly Detection algorithm using variational autoencoders.
    Arguments:
        input_enc: dimension of the encoder
        random_state: random number generator seed.
  """
  def __init__(self, input_size, input_enc, encoder = None, decoder = None, 
          layer=tf.keras.layers.LeakyReLU()):
    super(VariationalAutoencoder, self).__init__()

    #regularizer = None
    regularizer = tf.keras.regularizers.l1(10e-5)

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
          keras.layers.Dense(input_size, activation="sigmoid", activity_regularizer=regularizer)])
    else:
        self.decoder = decoder 

    self.flatten = keras.layers.Flatten()
    self.dense = keras.layers.Dense(input_enc + input_enc)

    self.encoder_full = lambda x: self.dense(self.flatten(x))

    self.sampling = Sampling()

    self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    self.variational_error_tracker = keras.metrics.Mean(
        name="variational_error"
    )

   
  @property
  def metrics(self):
      return [
          self.total_loss_tracker,
          self.variational_error_tracker,
      ]


  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)


  def encode(self, enc):
    mean, logvar = tf.split(self.encoder_full(enc), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

  def compute_errors(self, X, reconstruction, mean, log_var, z):    
    print("train_step: variational_error")
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=X)

    print(f"train_step: logpx_z {cross_ent}")
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
    print("train_step: logpz")
    logpz = log_normal_pdf(z, 0., 0.)
    print(f"train_step: logqz_x z {z} mean {mean} log_var {log_var}")
    logqz_x = log_normal_pdf(z, mean, log_var)
     
    print("train_step: reduce_mean")
    variational_loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

    print("train_step: total_loss")
    total_loss = variational_loss 

    return variational_loss, total_loss 

  def add_losses(self, variational_error, total_loss):
    self.total_loss_tracker.update_state(total_loss)
    self.variational_error_tracker.update_state(variational_error)

  def call(self, data):
      X, y = data

      encoder = self.encoder(X)

      print("call:encode")
      mean, log_var = self.encode(encoder)

      print(f"call:reparameterize mean {mean} log_var {log_var}")
      z = self.sampling([mean, log_var])
 
      print("call: decode")
      reconstruction = self.decode(z)

      return reconstruction, mean, log_var, z
      
  def test_step(self, data):
    X, y = data

    # Compute predictions
    probs, reconstruction, mean, log_var, z = self(data, training=False)

    print(f"test_step: compute_errors probs {probs} reconstruction {reconstruction} mean {mean} log_var {log_var}")
    variational_error, total_loss = self.compute_errors(X, reconstruction, mean, log_var, z)

    self.add_losses(variational_error, total_loss)

    return {m.name: m.result() for m in self.metrics}



  def train_step(self, data):
        print("data")
        X, y = data

        with tf.GradientTape() as tape:
            reconstruction, mean, log_var, z = self.call(data)
            variational_error, total_loss = self.compute_errors(X, reconstruction, mean, log_var, z)


        print("train_step: grads")
        grads = tape.gradient(total_loss, self.trainable_weights)
        print("train_step: apply_gradients")
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        print("train_step: loss_tracker")
        self.add_losses(variational_error, total_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "variational_error": self.variational_error_tracker.result()
        }

