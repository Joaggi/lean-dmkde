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

class Qadvaeff(keras.Model):
  """
    A Quantum Anomaly Detection algorithm using density matrices, Fourier features and variational autoencoders.
    Arguments:
        input_enc: dimension of the encoder
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output representation
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x * dim_y
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
  """
  def __init__(self, input_size, input_enc, dim_x, num_eig=0, gamma=1, random_state=None):
    super(Qadvaeff, self).__init__()

    self.encoder_1 = keras.layers.Dense(32, activation="relu") 
    self.encoder_2 = keras.layers.Dense(16, activation="relu") 
    self.encoder_3 = keras.layers.Dense(input_enc, activation="relu") 
    self.flatten = keras.layers.Flatten()
    self.dense = keras.layers.Dense(input_enc + input_enc)

    self.sampling = Sampling()

    self.encoder = lambda x: self.dense(self.flatten(self.encoder_3(self.encoder_2(self.encoder_1(x)))))


    self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_enc,
            dim=dim_x, gamma=gamma, random_state=random_state)
    self.fm_x.trainable = False
    self.qmd = layers.QMeasureDensityEig(dim_x=dim_x, num_eig=num_eig)
    self.num_eig = num_eig
    self.dim_x = dim_x
    self.gamma = gamma
    self.random_state = random_state

    self.decoder_1 = keras.layers.Dense(16, activation="relu")
    self.decoder_2 = keras.layers.Dense(32, activation="relu")
    self.decoder_3 = keras.layers.Dense(input_size, activation="sigmoid")

    self.decoder = lambda x: self.decoder_3(self.decoder_2(self.decoder_1(x)))

    self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    self.variational_error_tracker = keras.metrics.Mean(
        name="variational_error"
    )
    self.probs_loss_tracker = keras.metrics.Mean(name="probs_loss")

   
  @property
  def metrics(self):
      return [
          self.total_loss_tracker,
          self.variational_error_tracker,
          self.probs_loss_tracker
      ]


  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
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

  def compute_errors(self, X, probs, reconstruction, mean, log_var, z):    
    print("train_step: probs_loss")
    probs_loss = -tf.reduce_sum(tf.math.log(probs))

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
    total_loss = variational_loss + probs_loss

    return probs_loss, variational_loss, total_loss 

  def add_losses(self, probs_loss, variational_error, total_loss):
    self.total_loss_tracker.update_state(total_loss)
    self.variational_error_tracker.update_state(variational_error)
    self.probs_loss_tracker.update_state(probs_loss)

  def call(self, data):
      X, y = data

      print("call:encode")
      mean, log_var = self.encode(X)
        
      print(f"call:reparameterize mean {mean} log_var {log_var}")
      z = self.sampling([mean, log_var])

      print(f"call: fm_x z {z}")
      rff = self.fm_x(z)

      print("call: qmd")
      probs = self.qmd(rff)
      
      print("call: decode")
      reconstruction = self.decode(z)

      return probs, reconstruction, mean, log_var, z
      
  def test_step(self, data):
    X, y = data

    # Compute predictions
    probs, reconstruction, mean, log_var, z = self(data, training=False)

    print(f"test_step: compute_errors probs {probs} reconstruction {reconstruction} mean {mean} log_var {log_var}")
    probs_loss, variational_error, total_loss = self.compute_errors(X, probs, reconstruction, mean, log_var, z)

    self.add_losses(probs_loss, variational_error, total_loss)

    return {m.name: m.result() for m in self.metrics}



  def train_step(self, data):
        print("data")
        X, y = data

        with tf.GradientTape() as tape:
            probs, reconstruction, mean, log_var, z = self.call(data)
            probs_loss, variational_error, total_loss = self.compute_errors(X, probs, reconstruction, mean, log_var, z)


        print("train_step: grads")
        grads = tape.gradient(total_loss, self.trainable_weights)
        print("train_step: apply_gradients")
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        print("train_step: loss_tracker")
        self.add_losses(probs_loss, variational_error, total_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "variational_error": self.variational_error_tracker.result(),
            "probs_loss": self.probs_loss_tracker.result(),
        }

